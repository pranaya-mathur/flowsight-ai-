# FlowSight AI

ML system for predicting shipment delays in supply chain operations. Predicts delay probability, estimates delay duration, and identifies likely causes with LLM-generated explanations.

> **Status**: In development. Core functionality works, but still rough around the edges (see Known Issues below).

## What It Does

- **Binary classification**: Will this shipment delay? (85% accuracy)
- **Regression**: How many days late? (MAE 1.24 days)
- **Multi-class**: Why will it delay? (78% accuracy)
- **Enrichment**: Adjusts predictions using vendor reliability and route patterns
- **Explanations**: Generates plain-English explanations via Groq LLaMA

## Quick Start

```bash
# Clone repo
git clone https://github.com/pranaya-mathur/flowsight-ai-.git
cd flowsight-ai-

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional - for LLM explanations)
echo "GROQ_API_KEY=your_key_here" > .env

# Initialize feature store (first time only)
python -c "from src.data.feature_store import FeatureStore; fs = FeatureStore(); fs.load_raw_tables()"

# Train models (if not already trained)
python -m src.models.train_ensemble
python -m src.models.train_delay_regressor
python -m src.models.train_delay_reason

# Start API
uvicorn src.api.main:app --reload --port 8000

# In another terminal, launch dashboard
streamlit run src/dashboard/app.py
```

Open http://localhost:8501 for the dashboard.

## Architecture

```
src/
├── data/               # DuckDB feature store + data loading
├── enrichment/         # Vendor/route adjustment layers
├── models/             # Training scripts (ensemble, regressor, classifier)
├── api/                # FastAPI backend
├── dashboard/          # Streamlit UI
└── explainability/     # LLM integration for explanations
```

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "origin_city": "Mumbai",
    "destination_city": "Delhi",
    "supplier_name": "V001",
    "product_category": "Electronics",
    "weight_kg": 500,
    "value_inr": 50000,
    "apply_enrichment": true,
    "explain": false
  }'
```

See [docs/API.md](docs/API.md) for full documentation.

## Performance

**Models** (see [metrics/model_performance.json](metrics/model_performance.json)):
- Ensemble binary classifier: 84.7% accuracy, 0.891 AUC
- Delay days regressor: 1.24 MAE, 2.15 RMSE
- Delay reason classifier: 78.2% accuracy

**Inference**:
- Without LLM: ~85ms avg, ~142ms p95
- With LLM explanation: ~1.8s (Groq API latency)

**Enrichment Impact**:
- +3.2% accuracy improvement over base model
- Vendor adjustment: avg ±8.7%
- 50 vendors tracked, 2,449 routes

## Tech Stack

- **ML**: CatBoost + XGBoost + LightGBM ensemble
- **Feature Store**: DuckDB (embedded analytics DB)
- **API**: FastAPI + Pydantic
- **Dashboard**: Streamlit + Plotly
- **LLM**: Groq (LLaMA 3.3-70B) via LangChain

## Why Not Neural Networks?

Tried transformers and multi-task NNs first (see `src/models/train_multitask_transformer.py`). Results were terrible:
- Transformer: 61% accuracy vs 85% for ensemble
- Neural net: Overfitting badly on 15k samples
- Traditional ML just works better on tabular data with limited samples

Kept the transformer code for reference.

## Known Issues

- [ ] Dashboard crashes with null/empty vendor IDs ([#2](https://github.com/pranaya-mathur/flowsight-ai-/issues/2))
- [ ] LLM explanations timeout on slow networks (30s limit)
- [ ] No caching for DuckDB lookups - adds ~10ms per request ([#1](https://github.com/pranaya-mathur/flowsight-ai-/issues/1))
- [ ] LightGBM underperforms on routes with <100 samples ([#3](https://github.com/pranaya-mathur/flowsight-ai-/issues/3))
- [ ] No model versioning yet ([#4](https://github.com/pranaya-mathur/flowsight-ai-/issues/4))
- [ ] Test coverage is ~40% (need more tests)

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) if you hit issues.

## Project Structure

**Key files**:
- `src/enrichment/inference_pipeline.py` - Main prediction coordinator
- `src/enrichment/vendor_adjustment.py` - Vendor reliability layer
- `src/api/endpoints.py` - API route handlers
- `src/dashboard/app.py` - Streamlit UI

**Utilities**:
- `test_*.py` - Integration tests
- `inspect_*.py` - Data validation scripts
- `diagnostic_*.py` - Schema debugging tools

## Development

**Adding features**:
```python
# Edit src/data/features.py
# Rebuild feature store
python -c "from src.data.feature_store import FeatureStore; fs = FeatureStore(); fs.load_raw_tables()"
# Retrain models
python -m src.models.train_ensemble
```

**Testing API**:
```bash
python test_api.py
```

**Logs**:
Check `logs/` directory or set `LOG_LEVEL=DEBUG` in config.

## Roadmap

**Short term**:
- [ ] Fix dashboard validation bugs
- [ ] Add request caching
- [ ] Improve test coverage
- [ ] Docker setup

**Medium term**:
- [ ] Model versioning (DVC or MLflow)
- [ ] Batch prediction endpoint
- [ ] A/B testing framework
- [ ] Monitoring dashboard

**Long term**:
- [ ] Real-time model updates
- [ ] Multi-region deployment
- [ ] Mobile app?

## Data

Requires three CSV files in `data/raw/`:
- `supply_chain_15000_expanded.csv` - Main shipment data
- `vendor_performance_download.csv` - Vendor statistics
- `shipments_augmented.csv` - Route statistics

Model artifacts go in `models/` (generated during training, not in git).

## Contributing

PRs welcome! Areas that need help:
- Test coverage improvement
- Performance optimization
- Documentation
- Bug fixes (see issues)

## License

MIT

## Acknowledgments

- CatBoost team for the excellent docs
- Groq for fast LLM inference
- Supply chain dataset from [source TBD]
