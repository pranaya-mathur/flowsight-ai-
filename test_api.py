"""Test FastAPI endpoints."""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print(f"\n{'='*60}")
    print("HEALTH CHECK")
    print('='*60)
    print(json.dumps(response.json(), indent=2))


def test_prediction():
    """Test prediction endpoint."""
    payload = {
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
        "explain": False
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\n{'='*60}")
    print("PREDICTION TEST")
    print('='*60)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("\nTesting FlowSight API...")
    
    try:
        test_health()
        test_prediction()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
