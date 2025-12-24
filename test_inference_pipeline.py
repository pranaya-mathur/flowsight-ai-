"""Test complete inference pipeline."""

from src.enrichment.inference_pipeline import FlowSightPredictor
from src.logging_utils import get_logger

logger = get_logger(__name__)


def test_complete_prediction():
    """Test end-to-end prediction."""
    print("\n" + "="*60)
    print("TESTING COMPLETE INFERENCE PIPELINE")
    print("="*60)
    
    # Initialize predictor
    predictor = FlowSightPredictor()
    
    # Sample shipment (use actual feature names from your data)
    sample_shipment = {
        # Categorical features
        'origin_city': 'Mumbai',
        'destination_city': 'Delhi',
        'origin_state': 'Maharashtra',
        'destination_state': 'Delhi',
        'product_category': 'Electronics',
        'supplier_name': 'V001',
        'carrier_name': 'BlueDart',
        'truck_type': 'MHCV',
        'month': 'June',
        
        # Numeric features
        'quantity': 100,
        'weight_kg': 500.0,
        'value_inr': 50000.0,
        'risk_score': 0.6,
        
        # Enrichment features (will be filled if missing)
        'vendor_avg_delay_days': 1.45,
        'vendor_on_time_rate': 0.548,
        'route_distance_km': 1400.0,
        'route_vendor_reliability_sim': 0.7,
        'route_delay_probability_sim': 0.35,
        
        # Derived features
        'value_per_kg': 100.0,
        'route_pair': 'Mumbai->Delhi'
    }
    
    # Make prediction
    print("\nMaking prediction...")
    result = predictor.predict(sample_shipment, apply_enrichment=True)
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print('='*60)
    print(f"Will Delay: {'YES' if result['will_delay'] else 'NO'}")
    print(f"Delay Probability: {result['delay_probability']:.1%}")
    print(f"  - Raw Model: {result['delay_probability_raw']:.1%}")
    print(f"  - After Enrichment: {result['delay_probability']:.1%}")
    print(f"  - Enrichment Impact: {(result['delay_probability'] - result['delay_probability_raw'])*100:+.1f}%")
    print(f"\nEstimated Delay: {result['estimated_delay_days']:.1f} days")
    print(f"Likely Reason: {result['delay_reason']}")
    
    if result['delay_reason_confidence']:
        print(f"\nTop Delay Reasons:")
        for reason, conf in result['delay_reason_confidence'].items():
            print(f"  - {reason}: {conf:.1%}")
    
    # Get explanation context
    print(f"\n{'='*60}")
    print("EXPLANATION CONTEXT")
    print('='*60)
    context = predictor.get_explanation_context(sample_shipment)
    
    if context['vendor']['found']:
        print(f"\nVendor: {context['vendor']['vendor_id']}")
        print(f"  - Reliability: {context['vendor']['reliability_tier']}")
        print(f"  - On-time Rate: {context['vendor']['on_time_rate']:.1%}")
        print(f"  - Avg Delay: {context['vendor']['avg_delay_days']:.1f} days")
    
    if context['route']['found']:
        print(f"\nRoute: {context['route']['route']}")
        print(f"  - Historical Delay Rate: {context['route']['historical_delay_rate']:.1%}")
        print(f"  - Confidence: {context['route']['confidence']:.1%}")
        print(f"  - Distance: {context['route']['avg_distance_km']:.0f} km")


if __name__ == "__main__":
    test_complete_prediction()
