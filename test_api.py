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
    """Test prediction endpoint without explanation."""
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
    print("PREDICTION TEST (No Explanation)")
    print('='*60)
    print(json.dumps(response.json(), indent=2))


def test_prediction_with_explanation():
    """Test prediction with LLM explanation."""
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
        "explain": True  # Enable LLM explanation
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    result = response.json()
    
    print(f"\n{'='*60}")
    print("PREDICTION WITH LLM EXPLANATION")
    print('='*60)
    
    print(f"\n📊 PREDICTION:")
    print(json.dumps(result['prediction'], indent=2))
    
    print(f"\n🔧 ENRICHMENT:")
    print(json.dumps(result['enrichment'], indent=2))
    
    print(f"\n💡 LLM EXPLANATION:")
    print('='*60)
    print(result['explanation'])
    print('='*60)


if __name__ == "__main__":
    print("\n🚀 Testing FlowSight API...")
    
    try:
        test_health()
        test_prediction()
        test_prediction_with_explanation()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
