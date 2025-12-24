"""Pydantic schemas for request/response validation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ShipmentRequest(BaseModel):
    """Request schema for shipment prediction."""
    
    # Categorical features
    origin_city: str = Field(..., description="Origin city name")
    destination_city: str = Field(..., description="Destination city name")
    origin_state: str = Field(..., description="Origin state")
    destination_state: str = Field(..., description="Destination state")
    product_category: str = Field(..., description="Product category")
    supplier_name: str = Field(..., description="Vendor/supplier ID")
    carrier_name: str = Field(..., description="Carrier/logistics provider")
    truck_type: str = Field(..., description="Vehicle type (LCV/MHCV/Trailer)")
    month: str = Field(..., description="Shipping month")
    
    # Numeric features
    quantity: int = Field(..., gt=0, description="Order quantity")
    weight_kg: float = Field(..., gt=0, description="Shipment weight in kg")
    value_inr: float = Field(..., gt=0, description="Order value in INR")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    
    # Optional enrichment features (auto-filled if missing)
    vendor_avg_delay_days: Optional[float] = Field(None, description="Vendor avg delay")
    vendor_on_time_rate: Optional[float] = Field(None, description="Vendor OTR")
    route_distance_km: Optional[float] = Field(None, description="Route distance")
    route_vendor_reliability_sim: Optional[float] = Field(None, description="Route reliability")
    route_delay_probability_sim: Optional[float] = Field(None, description="Route delay prob")
    
    # Options
    apply_enrichment: bool = Field(True, description="Apply vendor/route enrichment")
    explain: bool = Field(False, description="Include LLM explanation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "origin_city": "Mumbai",
                "destination_city": "Delhi",
                "origin_state": "Maharashtra",
                "destination_state": "Delhi",
                "product_category": "Electronics",
                "supplier_name": "V001",
                "carrier_name": "BlueDart",
                "truck_type": "MHCV",
                "month": "June",
                "quantity": 100,
                "weight_kg": 500.0,
                "value_inr": 50000.0,
                "risk_score": 0.6,
                "apply_enrichment": True,
                "explain": True
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    
    prediction: Dict = Field(..., description="Core prediction results")
    enrichment: Dict = Field(..., description="Enrichment layer details")
    explanation: Optional[str] = Field(None, description="Natural language explanation")
    metadata: Dict = Field(..., description="Request metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": {
                    "will_delay": True,
                    "delay_probability": 0.72,
                    "estimated_delay_days": 2.3,
                    "delay_reason": "Vendor Capacity Issue",
                    "confidence": 0.85
                },
                "enrichment": {
                    "raw_model_probability": 0.65,
                    "vendor_adjustment": 0.07,
                    "route_adjustment": 0.0,
                    "vendor_tier": "POOR",
                    "route_confidence": 0.44
                },
                "explanation": "High delay risk due to poor vendor reliability...",
                "metadata": {
                    "model_version": "v2.0",
                    "timestamp": "2025-12-24T16:30:00Z"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    models_loaded: bool
    enrichment_enabled: bool
    vendor_count: int
    route_count: int
