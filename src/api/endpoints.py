"""API endpoint implementations."""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.api.schemas import ShipmentRequest, PredictionResponse, HealthResponse
from src.enrichment.inference_pipeline import FlowSightPredictor
from src.exceptions import ModelInferenceError
from src.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global predictor instance (loaded once at startup)
_predictor: Optional[FlowSightPredictor] = None


def get_predictor() -> FlowSightPredictor:
    """Dependency to get predictor instance."""
    global _predictor
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    return _predictor


def initialize_predictor():
    """Initialize predictor at startup."""
    global _predictor
    try:
        logger.info("Initializing FlowSight predictor...")
        _predictor = FlowSightPredictor()
        logger.info("Predictor initialized successfully")
    except Exception as exc:
        logger.exception("Failed to initialize predictor")
        raise RuntimeError("Predictor initialization failed") from exc


@router.get("/health", response_model=HealthResponse)
async def health_check(predictor: FlowSightPredictor = Depends(get_predictor)):
    """Check API health status."""
    try:
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            enrichment_enabled=True,
            vendor_count=len(predictor.vendor_enrichment.vendor_map),
            route_count=len(predictor.route_validation.route_map)
        )
    except Exception as exc:
        logger.exception("Health check failed")
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            enrichment_enabled=False,
            vendor_count=0,
            route_count=0
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_delay(
    request: ShipmentRequest,
    predictor: FlowSightPredictor = Depends(get_predictor)
):
    """
    Predict shipment delay with enrichment and optional explanation.
    
    - **origin_city**: Shipment origin
    - **destination_city**: Shipment destination
    - **supplier_name**: Vendor/supplier ID
    - **apply_enrichment**: Enable vendor/route adjustments
    - **explain**: Include natural language explanation
    """
    try:
        # Convert request to dict
        shipment_data = request.model_dump()
        
        # Extract options
        apply_enrichment = shipment_data.pop('apply_enrichment', True)
        explain = shipment_data.pop('explain', False)
        
        # Add derived features
        shipment_data['value_per_kg'] = shipment_data['value_inr'] / max(shipment_data['weight_kg'], 1)
        shipment_data['route_pair'] = f"{shipment_data['origin_city']}->{shipment_data['destination_city']}"
        
        # Make prediction
        result = predictor.predict(shipment_data, apply_enrichment=apply_enrichment)
        
        # Get enrichment context
        context = predictor.get_explanation_context(shipment_data)
        
        # Build response - CONVERT ALL NUMPY TYPES TO PYTHON TYPES
        response_data = {
            "prediction": {
                "will_delay": bool(result['will_delay']),  # Convert numpy.bool
                "delay_probability": float(round(result['delay_probability'], 3)),
                "estimated_delay_days": float(round(result['estimated_delay_days'], 2)),
                "delay_reason": str(result['delay_reason']),
                "confidence": float(max(result.get('delay_reason_confidence', {}).values())) if result.get('delay_reason_confidence') else 0.0
            },
            "enrichment": {
                "raw_model_probability": float(round(result['delay_probability_raw'], 3)),
                "vendor_adjustment": float(round(result['delay_probability'] - result['delay_probability_raw'], 3)),
                "route_adjustment": 0.0,
                "vendor_tier": str(context['vendor'].get('reliability_tier', 'UNKNOWN')),
                "route_confidence": float(round(context['route'].get('confidence', 0.0), 3)),
                "vendor_on_time_rate": float(round(context['vendor'].get('on_time_rate', 0.0), 3)),
                "route_historical_delay": float(round(context['route'].get('historical_delay_rate', 0.0), 3))
            },
            "explanation": None,
            "metadata": {
                "model_version": "v2.0",
                "timestamp": datetime.utcnow().isoformat(),
                "enrichment_applied": bool(result['enrichment_applied']),
                "route": str(result['metadata']['route']),
                "vendor": str(result['metadata']['vendor'])
            }
        }
        
        # Add LLM explanation if requested
        if explain:
            response_data["explanation"] = await generate_llm_explanation(
                result, context, shipment_data
            )
        
        return PredictionResponse(**response_data)
        
    except ModelInferenceError as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")
    except Exception as exc:
        logger.exception("Unexpected error in prediction endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")



async def generate_llm_explanation(result: dict, context: dict, shipment_data: dict) -> str:
    """Generate natural language explanation using LLM (placeholder for now)."""
    # We'll implement this in Part 2 with Groq LLM
    return "LLM explanation will be added in next step"


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FlowSight AI",
        "version": "2.0.0",
        "description": "Supply Chain Delay Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }
