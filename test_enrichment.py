"""Test enrichment layers with sample data."""

from src.enrichment.vendor_adjustment import VendorEnrichment
from src.enrichment.route_validation import RouteValidation
from src.logging_utils import get_logger

logger = get_logger(__name__)


def test_vendor_enrichment():
    """Test vendor adjustment logic."""
    print("\n" + "="*60)
    print("TESTING VENDOR ENRICHMENT")
    print("="*60)
    
    ve = VendorEnrichment()
    
    # Test scenarios
    scenarios = [
        ("V001", 0.5, "Average vendor"),  # Replace with actual vendor IDs
        ("V025", 0.7, "Good vendor"),
        ("V050", 0.3, "Poor vendor"),
    ]
    
    base_prob = 0.60  # 60% delay probability from model
    
    for vendor_id, expected_otr, description in scenarios:
        adjusted = ve.adjust_prediction(base_prob, vendor_id)
        context = ve.get_vendor_context(vendor_id)
        
        print(f"\n{description} ({vendor_id}):")
        print(f"  Base probability: {base_prob:.2%}")
        print(f"  Adjusted: {adjusted:.2%}")
        print(f"  On-time rate: {context.get('on_time_rate', 'N/A')}")
        print(f"  Tier: {context.get('reliability_tier', 'N/A')}")


def test_route_validation():
    """Test route validation logic."""
    print("\n" + "="*60)
    print("TESTING ROUTE VALIDATION")
    print("="*60)
    
    rv = RouteValidation()
    
    # Test scenarios (use actual cities from your data)
    scenarios = [
        ("Mumbai", "Delhi", 0.65, "Major route"),
        ("Kolkata", "Chennai", 0.45, "Secondary route"),
        ("UnknownCity", "AnotherCity", 0.50, "Unknown route"),
    ]
    
    for origin, dest, model_prob, description in scenarios:
        blended, confidence = rv.validate_prediction(model_prob, origin, dest)
        context = rv.get_route_context(origin, dest)
        
        print(f"\n{description} ({origin} -> {dest}):")
        print(f"  Model probability: {model_prob:.2%}")
        print(f"  Blended: {blended:.2%}")
        print(f"  Confidence: {confidence:.2%}")
        if context['found']:
            print(f"  Historical rate: {context['historical_delay_rate']:.2%}")
            print(f"  Sample size: {context['sample_size']}")


def test_combined_enrichment():
    """Test full enrichment pipeline."""
    print("\n" + "="*60)
    print("TESTING COMBINED ENRICHMENT")
    print("="*60)
    
    ve = VendorEnrichment()
    rv = RouteValidation()
    
    # Simulate a prediction
    model_prob = 0.55
    vendor_id = "V001"  # Replace with actual
    origin = "Mumbai"   # Replace with actual
    destination = "Delhi"  # Replace with actual
    
    print(f"\nInitial model prediction: {model_prob:.2%}")
    
    # Step 1: Vendor adjustment
    vendor_adjusted = ve.adjust_prediction(model_prob, vendor_id)
    print(f"After vendor adjustment: {vendor_adjusted:.2%}")
    
    # Step 2: Route validation
    final_prob, confidence = rv.validate_prediction(vendor_adjusted, origin, destination)
    print(f"After route validation: {final_prob:.2%}")
    print(f"Route confidence: {confidence:.2%}")
    
    print(f"\n Final prediction: {final_prob:.2%} (from {model_prob:.2%})")


if __name__ == "__main__":
    test_vendor_enrichment()
    test_route_validation()
    test_combined_enrichment()
