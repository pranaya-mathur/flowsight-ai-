# API Documentation

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

**GET** `/health`

Check API and model status.

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "enrichment_enabled": true,
  "vendor_count": 50,
  "route_count": 2449
}
```

### Predict Delay

**POST** `/predict`

Predict shipment delay with optional enrichment and explanation.

**Request Body**:
```json
{
  "origin_city": "Mumbai",
  "destination_city": "Delhi",
  "origin_state": "Maharashtra",
  "destination_state": "Delhi",
  "product_category": "Electronics",
  "supplier_name": "V001",
  "carrier_name": "BlueDart",
  "truck_type": "LCV",
  "month": "January",
  "quantity": 100,
  "weight_kg": 500.0,
  "value_inr": 50000.0,
  "risk_score": 0.6,
  "apply_enrichment": true,
  "explain": true
}
```

**Example with curl**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin_city": "Mumbai",
    "destination_city": "Delhi",
    "origin_state": "Maharashtra",
    "destination_state": "Delhi",
    "product_category": "Electronics",
    "supplier_name": "V001",
    "carrier_name": "BlueDart",
    "truck_type": "LCV",
    "month": "January",
    "quantity": 100,
    "weight_kg": 500.0,
    "value_inr": 50000.0,
    "risk_score": 0.6,
    "apply_enrichment": true,
    "explain": false
  }'
```

**Response**:
```json
{
  "prediction": {
    "will_delay": true,
    "delay_probability": 0.782,
    "estimated_delay_days": 3.2,
    "delay_reason": "vendor_issues",
    "confidence": 0.68
  },
  "enrichment": {
    "raw_model_probability": 0.651,
    "vendor_adjustment": 0.131,
    "route_adjustment": 0.0,
    "vendor_tier": "POOR",
    "route_confidence": 0.89,
    "vendor_on_time_rate": 0.45,
    "route_historical_delay": 0.64
  },
  "explanation": "This shipment has a 78% chance of delay...",
  "metadata": {
    "model_version": "v2.0",
    "timestamp": "2026-01-17T04:55:12.123Z",
    "enrichment_applied": true,
    "route": "Mumbai->Delhi",
    "vendor": "V001"
  }
}
```

## Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

def predict_shipment_delay(shipment_data):
    """Make delay prediction for a shipment."""
    response = requests.post(
        f"{API_URL}/predict",
        json=shipment_data,
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.text}")

# Example usage
shipment = {
    "origin_city": "Bangalore",
    "destination_city": "Chennai",
    "origin_state": "Karnataka",
    "destination_state": "Tamil Nadu",
    "product_category": "FMCG",
    "supplier_name": "V015",
    "carrier_name": "DTDC",
    "truck_type": "MHCV",
    "month": "March",
    "quantity": 250,
    "weight_kg": 1200.0,
    "value_inr": 75000.0,
    "risk_score": 0.4,
    "apply_enrichment": True,
    "explain": False  # Faster without LLM
}

result = predict_shipment_delay(shipment)
print(f"Delay probability: {result['prediction']['delay_probability']:.1%}")
print(f"Vendor tier: {result['enrichment']['vendor_tier']}")
```

## Parameters

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|----------|
| `origin_city` | string | Origin city name | "Mumbai" |
| `destination_city` | string | Destination city | "Delhi" |
| `origin_state` | string | Origin state | "Maharashtra" |
| `destination_state` | string | Destination state | "Delhi" |
| `product_category` | string | Product type | "Electronics" |
| `supplier_name` | string | Vendor/supplier ID | "V001" |
| `carrier_name` | string | Logistics carrier | "BlueDart" |
| `truck_type` | string | Vehicle type | "LCV", "MHCV", "Trailer" |
| `month` | string | Shipment month | "January" |
| `quantity` | integer | Item quantity | 100 |
| `weight_kg` | float | Weight in kg | 500.0 |
| `value_inr` | float | Value in INR | 50000.0 |
| `risk_score` | float | Risk score (0-1) | 0.6 |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `apply_enrichment` | boolean | true | Enable vendor/route adjustments |
| `explain` | boolean | false | Generate LLM explanation (slower) |

## Response Fields

### prediction

- `will_delay`: Boolean, true if delay probability ≥ 0.5
- `delay_probability`: Float (0-1), final probability after enrichment
- `estimated_delay_days`: Float, predicted delay duration
- `delay_reason`: String, primary cause ("vendor_issues", "weather_disruption", etc.)
- `confidence`: Float (0-1), model confidence in delay reason

### enrichment

- `raw_model_probability`: Float, base model prediction before adjustments
- `vendor_adjustment`: Float, change from vendor enrichment layer
- `vendor_tier`: String, vendor reliability ("EXCELLENT", "GOOD", "AVERAGE", "POOR")
- `vendor_on_time_rate`: Float, vendor's historical on-time delivery rate
- `route_historical_delay`: Float, route's historical delay rate
- `route_confidence`: Float, confidence in route statistics

## Error Responses

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "weight_kg"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "detail": "Prediction failed: Model inference error"
}
```

### 503 Service Unavailable

```json
{
  "detail": "Predictor not initialized"
}
```

## Performance Notes

- Average response time: **50-100ms** (without LLM)
- With LLM explanation: **1-2 seconds** (depends on Groq API)
- Model loading at startup: **~2 seconds**
- Feature store lookup: **<10ms**

## Rate Limiting

Currently no rate limiting implemented. For production use, consider adding:
- Rate limiting middleware
- Request queuing for high load
- Caching for repeated predictions
