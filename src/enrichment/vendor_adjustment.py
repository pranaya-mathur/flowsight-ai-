"""
Vendor-based prediction adjustment.

Adjusts delay probability based on vendor historical performance:
- Poor vendors (on_time_rate < 50%): +15% delay probability
- Good vendors (on_time_rate > 65%): -10% delay probability
- Average vendors: minimal adjustment
"""

from typing import Optional
import duckdb
import numpy as np
from pathlib import Path

from config import settings
from src.exceptions import EnrichmentError
from src.logging_utils import get_logger

logger = get_logger(__name__)


class VendorEnrichment:
    """Adjust predictions using vendor performance statistics."""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize vendor enrichment with DuckDB connection.
        
        Args:
            db_path: Path to feature store database. Uses default if None.
        """
        try:
            self.db_path = db_path or (settings.FEATURE_STORE_DIR / "flowsight.duckdb")
            self.con = duckdb.connect(str(self.db_path), read_only=True)
            self._load_vendor_stats()
            logger.info("Vendor enrichment initialized with %d vendors", len(self.vendor_map))
        except Exception as exc:
            logger.exception("Failed to initialize vendor enrichment")
            raise EnrichmentError("Could not initialize vendor enrichment") from exc
    
    def _load_vendor_stats(self) -> None:
        """Load vendor statistics into memory for fast lookup."""
        # TODO: Add caching mechanism to avoid reloading on every initialization
        # TODO: Consider using Redis for distributed deployments
        try:
            query = """
            SELECT 
                vendor_id,
                on_time_rate,
                avg_delay_days,
                shipments
            FROM vendor_stats
            """
            df = self.con.execute(query).fetch_df()
            
            # Create lookup dictionary
            self.vendor_map = {}
            for _, row in df.iterrows():
                self.vendor_map[row['vendor_id']] = {
                    'on_time_rate': row['on_time_rate'],
                    'avg_delay_days': row['avg_delay_days'],
                    'total_shipments': row['shipments']
                }
            
            logger.info("Loaded %d vendor profiles", len(self.vendor_map))
            
        except Exception as exc:
            logger.exception("Failed to load vendor statistics")
            raise EnrichmentError("Failed to load vendor stats from feature store") from exc
    
    def adjust_prediction(
        self,
        base_probability: float,
        vendor_id: str,
        confidence_weight: float = 1.0
    ) -> float:
        """
        Adjust delay probability based on vendor performance.
        
        Args:
            base_probability: Model's raw delay probability (0-1)
            vendor_id: Vendor/supplier identifier
            confidence_weight: How much to trust vendor adjustment (0-1)
        
        Returns:
            Adjusted delay probability, clipped to [0, 1]
        """
        try:
            # FIXME: Handle edge case when vendor_id is None or empty string
            if vendor_id not in self.vendor_map:
                logger.warning("Unknown vendor: %s, using base prediction", vendor_id)
                return base_probability
            
            vendor_stats = self.vendor_map[vendor_id]
            on_time_rate = vendor_stats['on_time_rate']
            
            # Calculate adjustment based on vendor reliability
            # NOTE: These thresholds (0.50, 0.65) were determined empirically
            # TODO: Consider A/B testing different threshold values
            adjustment = self._calculate_adjustment(on_time_rate)
            
            # Apply confidence weighting
            weighted_adjustment = adjustment * confidence_weight
            
            # Adjust prediction
            adjusted_prob = base_probability + weighted_adjustment
            
            # Clip to valid probability range
            adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)
            
            logger.debug(
                "Vendor %s (OTR=%.2f): %.3f -> %.3f (adj=%.3f)",
                vendor_id, on_time_rate, base_probability, adjusted_prob, weighted_adjustment
            )
            
            return adjusted_prob
            
        except Exception as exc:
            logger.exception("Vendor adjustment failed for %s", vendor_id)
            # Fail gracefully - return original prediction
            return base_probability
    
    def _calculate_adjustment(self, on_time_rate: float) -> float:
        """
        Calculate probability adjustment based on vendor on-time rate.
        
        Adjustment strategy (as per plan):
        - Excellent (>65%): -10% delay probability
        - Poor (<50%): +15% delay probability  
        - Average (50-65%): linear interpolation
        
        Args:
            on_time_rate: Vendor's historical on-time rate (0-1)
        
        Returns:
            Adjustment value to add to probability (-0.15 to +0.10)
        """
        # TODO: These adjustment values are currently hardcoded
        # Consider making them configurable or learning them from data
        if on_time_rate >= 0.65:
            # Good vendor - reduce delay probability
            return -0.10
        
        elif on_time_rate <= 0.50:
            # Poor vendor - increase delay probability
            return 0.15
        
        else:
            # Average vendor - linear interpolation
            # HACK: Simple linear scaling for now, could use sigmoid or other curves
            normalized = (on_time_rate - 0.50) / (0.65 - 0.50)
            adjustment = 0.15 + normalized * (-0.10 - 0.15)
            return adjustment
    
    def get_vendor_context(self, vendor_id: str) -> dict:
        """
        Get vendor statistics for explanation/context.
        
        Args:
            vendor_id: Vendor identifier
        
        Returns:
            Dictionary with vendor stats, or None if vendor unknown
        """
        if vendor_id not in self.vendor_map:
            return {
                'found': False,
                'vendor_id': vendor_id,
                'message': 'Unknown vendor - no historical data'
            }
        
        stats = self.vendor_map[vendor_id]
        return {
            'found': True,
            'vendor_id': vendor_id,
            'on_time_rate': stats['on_time_rate'],
            'avg_delay_days': stats['avg_delay_days'],
            'total_shipments': stats['total_shipments'],
            'reliability_tier': self._get_reliability_tier(stats['on_time_rate'])
        }
    
    def _get_reliability_tier(self, on_time_rate: float) -> str:
        """Classify vendor reliability into tiers."""
        # NOTE: Tier boundaries chosen based on business requirements
        if on_time_rate >= 0.65:
            return 'EXCELLENT'
        elif on_time_rate >= 0.55:
            return 'GOOD'
        elif on_time_rate >= 0.45:
            return 'AVERAGE'
        else:
            return 'POOR'
    
    def __del__(self):
        """Close database connection on cleanup."""
        if hasattr(self, 'con'):
            self.con.close()
