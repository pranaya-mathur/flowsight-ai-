"""
Route-based prediction validation.

Blends model predictions with historical route statistics using
confidence-based weighting:
- High confidence routes: 80% model, 20% history
- Low confidence routes: 50% model, 50% history
"""

from typing import Optional, Tuple
import duckdb
import numpy as np
from pathlib import Path

from config import settings
from src.exceptions import EnrichmentError
from src.logging_utils import get_logger

logger = get_logger(__name__)


class RouteValidation:
    """Validate predictions against historical route patterns."""
    
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize route validation with DuckDB connection.
        
        Args:
            db_path: Path to feature store database. Uses default if None.
        """
        try:
            self.db_path = db_path or (settings.FEATURE_STORE_DIR / "flowsight.duckdb")
            self.con = duckdb.connect(str(self.db_path), read_only=True)
            self._load_route_stats()
            logger.info("Route validation initialized with %d routes", len(self.route_map))
        except Exception as exc:
            logger.exception("Failed to initialize route validation")
            raise EnrichmentError("Could not initialize route validation") from exc
    
    def _load_route_stats(self) -> None:
        """Load route statistics into memory for fast lookup."""
        try:
            query = """
            SELECT 
                origin,
                destination,
                distance_km,
                delay_probability_sim,
                vendor_reliability_sim,
                lateness_flag
            FROM route_stats
            WHERE origin IS NOT NULL 
              AND destination IS NOT NULL
            """
            df = self.con.execute(query).fetch_df()
            
            # Aggregate by route (origin -> destination)
            route_groups = df.groupby(['origin', 'destination']).agg({
                'delay_probability_sim': ['mean', 'std', 'count'],
                'distance_km': 'mean',
                'vendor_reliability_sim': 'mean',
                'lateness_flag': 'mean'  # Actual delay rate
            }).reset_index()
            
            # Flatten column names
            route_groups.columns = [
                'origin', 'destination',
                'avg_delay_prob', 'std_delay_prob', 'sample_count',
                'avg_distance', 'avg_reliability', 'actual_delay_rate'
            ]
            
            # Create lookup dictionary
            self.route_map = {}
            for _, row in route_groups.iterrows():
                route_key = f"{row['origin']}->{row['destination']}"
                self.route_map[route_key] = {
                    'avg_delay_prob': row['avg_delay_prob'],
                    'std_delay_prob': row['std_delay_prob'],
                    'sample_count': int(row['sample_count']),
                    'avg_distance': row['avg_distance'],
                    'avg_reliability': row['avg_reliability'],
                    'actual_delay_rate': row['actual_delay_rate']
                }
            
            logger.info("Loaded %d unique routes", len(self.route_map))
            
        except Exception as exc:
            logger.exception("Failed to load route statistics")
            raise EnrichmentError("Failed to load route stats from feature store") from exc
    
    def validate_prediction(
        self,
        model_probability: float,
        origin_city: str,
        destination_city: str
    ) -> Tuple[float, float]:
        """
        Blend model prediction with historical route data.
        
        Args:
            model_probability: Model's delay probability (0-1)
            origin_city: Shipment origin
            destination_city: Shipment destination
        
        Returns:
            Tuple of (blended_probability, confidence_score)
        """
        try:
            route_key = f"{origin_city}->{destination_city}"
            
            # Check if route exists in historical data
            if route_key not in self.route_map:
                logger.debug("Unknown route: %s, using model prediction", route_key)
                return model_probability, 0.0  # Zero confidence in history
            
            route_stats = self.route_map[route_key]
            historical_prob = route_stats['actual_delay_rate']  # Use actual delay rate
            sample_count = route_stats['sample_count']
            std_dev = route_stats['std_delay_prob']
            
            # Calculate confidence based on sample size and variance
            confidence = self._calculate_confidence(sample_count, std_dev)
            
            # Blend predictions based on confidence
            blended_prob = self._blend_predictions(
                model_probability,
                historical_prob,
                confidence
            )
            
            logger.debug(
                "Route %s: model=%.3f, history=%.3f (n=%d), blended=%.3f, conf=%.2f",
                route_key, model_probability, historical_prob,
                sample_count, blended_prob, confidence
            )
            
            return blended_prob, confidence
            
        except Exception as exc:
            logger.exception("Route validation failed for %s->%s", origin_city, destination_city)
            # Fail gracefully - return original prediction
            return model_probability, 0.0
    
    def _calculate_confidence(self, sample_count: int, std_dev: float) -> float:
        """
        Calculate confidence in historical route data.
        
        High confidence when:
        - Large sample size (many historical shipments)
        - Low variance (consistent delay patterns)
        
        Args:
            sample_count: Number of historical shipments on this route
            std_dev: Standard deviation of historical delay probabilities
        
        Returns:
            Confidence score (0-1)
        """
        # Sample size component (more samples = higher confidence)
        # Sigmoid function: reaches 0.8 at 50 samples
        sample_conf = 1.0 / (1.0 + np.exp(-0.1 * (sample_count - 50)))
        
        # Variance component (lower std = higher confidence)
        # High std (>0.3) reduces confidence
        variance_conf = max(0.0, 1.0 - (std_dev / 0.3))
        
        # Combined confidence (average of both)
        confidence = (sample_conf + variance_conf) / 2.0
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _blend_predictions(
        self,
        model_prob: float,
        historical_prob: float,
        confidence: float
    ) -> float:
        """
        Blend model and historical predictions based on confidence.
        
        Strategy (as per plan):
        - High confidence (>0.7): 80% model, 20% history
        - Medium confidence (0.4-0.7): Linear interpolation
        - Low confidence (<0.4): 50% model, 50% history
        
        Args:
            model_prob: Model prediction
            historical_prob: Historical average
            confidence: Confidence in historical data
        
        Returns:
            Blended probability
        """
        if confidence >= 0.7:
            # High confidence - trust model more
            model_weight = 0.8
        elif confidence <= 0.4:
            # Low confidence - equal weighting
            model_weight = 0.5
        else:
            # Medium confidence - linear interpolation
            # confidence 0.4 -> weight 0.5
            # confidence 0.7 -> weight 0.8
            model_weight = 0.5 + (confidence - 0.4) / (0.7 - 0.4) * 0.3
        
        history_weight = 1.0 - model_weight
        
        blended = model_weight * model_prob + history_weight * historical_prob
        
        return np.clip(blended, 0.0, 1.0)
    
    def get_route_context(self, origin_city: str, destination_city: str) -> dict:
        """
        Get route statistics for explanation/context.
        
        Args:
            origin_city: Origin city
            destination_city: Destination city
        
        Returns:
            Dictionary with route stats, or None if route unknown
        """
        route_key = f"{origin_city}->{destination_city}"
        
        if route_key not in self.route_map:
            return {
                'found': False,
                'route': route_key,
                'message': 'Unknown route - no historical data'
            }
        
        stats = self.route_map[route_key]
        confidence = self._calculate_confidence(stats['sample_count'], stats['std_delay_prob'])
        
        return {
            'found': True,
            'route': route_key,
            'historical_delay_rate': stats['actual_delay_rate'],
            'sample_size': stats['sample_count'],
            'consistency': max(0.0, 1.0 - stats['std_delay_prob']),  # Higher = more consistent
            'confidence': confidence,
            'avg_distance_km': stats['avg_distance']
        }
    
    def __del__(self):
        """Close database connection on cleanup."""
        if hasattr(self, 'con'):
            self.con.close()
