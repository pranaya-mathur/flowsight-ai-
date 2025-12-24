"""
Unified inference pipeline for FlowSight predictions.

Combines:
1. Ensemble binary classifier
2. Delay days regressor
3. Delay reason classifier
4. Vendor enrichment
5. Route validation
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from config import settings
from src.enrichment.vendor_adjustment import VendorEnrichment
from src.enrichment.route_validation import RouteValidation
from src.exceptions import ModelInferenceError
from src.logging_utils import get_logger

logger = get_logger(__name__)


class FlowSightPredictor:
    """Complete prediction pipeline with enrichment."""
    
    def __init__(
        self,
        ensemble_path: Optional[Path] = None,
        regressor_path: Optional[Path] = None,
        classifier_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize prediction pipeline with all models.
        
        Args:
            ensemble_path: Path to ensemble binary model
            regressor_path: Path to delay days regressor
            classifier_path: Path to delay reason classifier
        """
        try:
            # Model paths
            self.ensemble_path = ensemble_path or (settings.MODELS_DIR / "ensemble_binary_delay.pkl")
            self.regressor_path = regressor_path or (settings.MODELS_DIR / "delay_days_regressor.pkl")
            self.classifier_path = classifier_path or (settings.MODELS_DIR / "delay_reason_classifier_v2.pkl")
            
            # Load models
            self._load_models()
            
            # Initialize enrichment layers
            self.vendor_enrichment = VendorEnrichment()
            self.route_validation = RouteValidation()
            
            logger.info("FlowSight predictor initialized successfully")
            
        except Exception as exc:
            logger.exception("Failed to initialize predictor")
            raise ModelInferenceError("Could not initialize FlowSight predictor") from exc
    
    def _load_models(self) -> None:
        """Load all trained models."""
        try:
            # Load ensemble binary classifier
            logger.info("Loading ensemble model from %s", self.ensemble_path)
            ensemble_artifacts = joblib.load(self.ensemble_path)
            self.ensemble_preprocessor = ensemble_artifacts['preprocessor']
            self.ensemble_catboost = ensemble_artifacts['catboost']
            self.ensemble_lightgbm = ensemble_artifacts['lightgbm']
            self.ensemble_xgboost = ensemble_artifacts['xgboost']
            self.ensemble_features = ensemble_artifacts['feature_cols']
            
            # Load delay regressor
            logger.info("Loading regressor from %s", self.regressor_path)
            regressor_artifacts = joblib.load(self.regressor_path)
            self.regressor_pipeline = regressor_artifacts['pipeline']
            self.regressor_features = regressor_artifacts['feature_cols']
            
            # Load delay reason classifier
            logger.info("Loading classifier from %s", self.classifier_path)
            classifier_artifacts = joblib.load(self.classifier_path)
            self.classifier_models = classifier_artifacts['models']
            self.classifier_features = classifier_artifacts['feature_cols']
            self.classifier_cat_indices = classifier_artifacts['cat_feature_indices']
            self.idx_to_class = classifier_artifacts['idx_to_class']
            
            logger.info("All models loaded successfully")
            
        except FileNotFoundError as exc:
            logger.error("Model file not found: %s", exc)
            raise ModelInferenceError(f"Model file missing: {exc}") from exc
        except Exception as exc:
            logger.exception("Failed to load models")
            raise ModelInferenceError("Failed to load model artifacts") from exc
    
    def predict(
        self,
        shipment_data: Dict,
        apply_enrichment: bool = True
    ) -> Dict:
        """
        Make complete prediction for a shipment.
        
        Args:
            shipment_data: Dictionary with shipment features
            apply_enrichment: Whether to apply vendor/route enrichment
        
        Returns:
            Dictionary with all predictions and metadata
        """
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([shipment_data])
            
            # 1. Binary prediction (will delay?)
            binary_prob = self._predict_binary(df)
            
            # 2. Apply enrichment if enabled
            if apply_enrichment:
                enriched_prob = self._apply_enrichment(
                    binary_prob,
                    shipment_data.get('supplier_name'),
                    shipment_data.get('origin_city'),
                    shipment_data.get('destination_city')
                )
            else:
                enriched_prob = binary_prob
            
            # 3. Regression prediction (how many days?)
            if enriched_prob >= 0.5:  # Predict delay magnitude only if likely to delay
                delay_days = self._predict_delay_days(df)
            else:
                delay_days = 0.0
            
            # 4. Classification prediction (why delay?)
            if enriched_prob >= 0.5:
                delay_reason, reason_probs = self._predict_delay_reason(df)
            else:
                delay_reason = "No Delay Expected"
                reason_probs = {}
            
            # Compile results
            result = {
                'will_delay': enriched_prob >= 0.5,
                'delay_probability': float(enriched_prob),
                'delay_probability_raw': float(binary_prob),
                'estimated_delay_days': float(delay_days),
                'delay_reason': delay_reason,
                'delay_reason_confidence': reason_probs,
                'enrichment_applied': apply_enrichment,
                'metadata': {
                    'vendor': shipment_data.get('supplier_name'),
                    'route': f"{shipment_data.get('origin_city')}->{shipment_data.get('destination_city')}",
                }
            }
            
            return result
            
        except Exception as exc:
            logger.exception("Prediction failed")
            raise ModelInferenceError("Prediction pipeline failed") from exc
    
    def _predict_binary(self, df: pd.DataFrame) -> float:
        """Predict delay probability using ensemble."""
        try:
            X = df[self.ensemble_features].copy()
            X_proc = self.ensemble_preprocessor.transform(X)
            
            # Get probabilities from all models
            cat_prob = self.ensemble_catboost.predict_proba(X_proc)[0, 1]
            lgbm_prob = self.ensemble_lightgbm.predict_proba(X_proc)[0, 1]
            xgb_prob = self.ensemble_xgboost.predict_proba(X_proc)[0, 1]
            
            # Ensemble average
            ensemble_prob = (cat_prob + lgbm_prob + xgb_prob) / 3.0
            
            return ensemble_prob
            
        except Exception as exc:
            logger.exception("Binary prediction failed")
            raise ModelInferenceError("Ensemble prediction failed") from exc
    
    def _predict_delay_days(self, df: pd.DataFrame) -> float:
        """Predict delay duration using regressor."""
        try:
            X = df[self.regressor_features].copy()
            days = self.regressor_pipeline.predict(X)[0]
            return max(0.0, days)  # Clip negative predictions
            
        except Exception as exc:
            logger.exception("Regression prediction failed")
            raise ModelInferenceError("Delay days prediction failed") from exc
    
    def _predict_delay_reason(self, df: pd.DataFrame) -> Tuple[str, Dict]:
        """Predict delay reason using classifier ensemble."""
        try:
            X = df[self.classifier_features].copy()
            
            # Get probabilities from all models in ensemble
            all_probs = []
            for model in self.classifier_models:
                probs = model.predict_proba(X)[0]
                all_probs.append(probs)
            
            # Average probabilities
            avg_probs = np.mean(all_probs, axis=0)
            
            # Get top prediction
            top_idx = np.argmax(avg_probs)
            top_reason = self.idx_to_class[top_idx]
            
            # Create confidence dict for top 3 reasons
            top_3_indices = np.argsort(avg_probs)[-3:][::-1]
            reason_probs = {
                self.idx_to_class[idx]: float(avg_probs[idx])
                for idx in top_3_indices
            }
            
            return top_reason, reason_probs
            
        except Exception as exc:
            logger.exception("Classification prediction failed")
            raise ModelInferenceError("Delay reason prediction failed") from exc
    
    def _apply_enrichment(
        self,
        base_probability: float,
        vendor_id: str,
        origin_city: str,
        destination_city: str
    ) -> float:
        """Apply vendor and route enrichment layers."""
        try:
            # Step 1: Vendor adjustment
            vendor_adjusted = self.vendor_enrichment.adjust_prediction(
                base_probability,
                vendor_id
            )
            
            # Step 2: Route validation
            route_adjusted, confidence = self.route_validation.validate_prediction(
                vendor_adjusted,
                origin_city,
                destination_city
            )
            
            logger.debug(
                "Enrichment: %.3f -> %.3f (vendor) -> %.3f (route, conf=%.2f)",
                base_probability, vendor_adjusted, route_adjusted, confidence
            )
            
            return route_adjusted
            
        except Exception as exc:
            logger.warning("Enrichment failed, using base prediction: %s", exc)
            return base_probability
    
    def get_explanation_context(self, shipment_data: Dict) -> Dict:
        """Get context for explainability (SHAP/LLM)."""
        vendor_context = self.vendor_enrichment.get_vendor_context(
            shipment_data.get('supplier_name')
        )
        route_context = self.route_validation.get_route_context(
            shipment_data.get('origin_city'),
            shipment_data.get('destination_city')
        )
        
        return {
            'vendor': vendor_context,
            'route': route_context
        }
