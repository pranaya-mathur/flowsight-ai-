# Troubleshooting Guide

Common issues encountered during development and deployment.

## API Issues

### Predictor Not Initialized Error

**Error**: `503 Service Unavailable - Predictor not initialized`

**Cause**: Models not loaded at startup or model files missing

**Solution**:
```bash
# Check if model files exist
ls -lh models/

# Should see:
# ensemble_binary_delay.pkl (~45MB)
# delay_days_regressor.pkl (~12MB)  
# delay_reason_classifier_v2.pkl (~18MB)

# If missing, train models first:
python -m src.models.train_ensemble
python -m src.models.train_delay_regressor
python -m src.models.train_delay_reason
```

### NaN in Prediction Response

**Symptom**: API returns `delay_probability: NaN`

**Cause**: Unknown vendor ID or route with insufficient data

**Debug**:
```python
# Check if vendor exists in feature store
from src.enrichment.vendor_adjustment import VendorEnrichment
ve = VendorEnrichment()
print(ve.vendor_map.get('YOUR_VENDOR_ID'))  # Should not be None
```

**Fix**: Add fallback handling for unknown vendors (already implemented in v0.2.0)

### LLM Timeout Errors

**Error**: `LLM explanation generation failed: timeout`

**Cause**: Groq API slow response or network issues

**Workaround**:
```python
# Disable explanations for faster responses
payload = {
    # ... other fields
    "explain": False  # Skip LLM call
}
```

## Dashboard Issues

### Dashboard Won't Start

**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
pip install streamlit plotly
```

### Connection Refused to API

**Symptom**: Dashboard shows "API Server is not running!"

**Check**:
```bash
# Ensure API is running on port 8000
curl http://localhost:8000/health

# If not running:
uvicorn src.api.main:app --reload --port 8000
```

### Slow Dashboard Performance

**Issue**: Predictions take 3-4 seconds

**Known causes**:
1. DuckDB connection overhead on each request
2. LLM API latency (can be 1-2s)
3. No caching implemented yet

**Temporary fix**: Disable LLM explanations for faster response

## Training Issues

### DuckDB Lock Error

**Error**: `database is locked`

**Cause**: Feature store open in another process

**Solution**:
```bash
# Check for processes using DuckDB
lsof data/feature_store/flowsight.duckdb

# Kill hanging processes if needed
```

### Feature Store Empty

**Error**: `No rows found in primary_shipments table`

**Solution**:
```python
# Rebuild feature store
from src.data.feature_store import FeatureStore

fs = FeatureStore()
fs.load_raw_tables()
fs.basic_health_check()  # Should return True
```

## Data Issues

### Missing CSV Files

**Error**: `FileNotFoundError: data/raw/supply_chain_15000_expanded.csv`

**Fix**: Ensure all raw data files are present:
- `data/raw/supply_chain_15000_expanded.csv` (primary dataset)
- `data/raw/vendor_performance_download.csv` (vendor stats)
- `data/raw/shipments_augmented.csv` (route stats)

### Vendor/Route Statistics Mismatch

**Issue**: Enrichment layer not adjusting predictions

**Debug**:
```python
# Check vendor stats loaded correctly
from src.enrichment.vendor_adjustment import VendorEnrichment

ve = VendorEnrichment()
print(f"Loaded {len(ve.vendor_map)} vendors")

# Should see ~50 vendors
# If 0, feature store needs rebuilding
```

## Environment Issues

### Groq API Key Not Found

**Warning**: `No Groq API key found. Explanations will be unavailable.`

**Fix**:
```bash
# Add to .env file
echo "GROQ_API_KEY=your_api_key_here" >> .env

# Restart API
```

## Performance Debugging

### Check Model Loading Time

```python
import time
from src.enrichment.inference_pipeline import FlowSightPredictor

start = time.time()
predictor = FlowSightPredictor()
print(f"Model loading: {time.time() - start:.2f}s")
# Should be < 2 seconds
```

### Check Prediction Latency

```python
test_shipment = {
    "origin_city": "Mumbai",
    "destination_city": "Delhi",
    "supplier_name": "V001",
    # ... other fields
}

start = time.time()
result = predictor.predict(test_shipment)
print(f"Prediction time: {time.time() - start:.3f}s")
# Should be < 0.1s without LLM
```

## Still Having Issues?

Check logs in `logs/` directory or enable debug logging:

```python
# config/settings.py
LOG_LEVEL = "DEBUG"  # Was INFO
```
