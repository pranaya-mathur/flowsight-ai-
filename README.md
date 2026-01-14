# FlowSight AI

Production-ready ML system for predicting shipment delays and estimating arrival times in supply chain operations. Built to handle real-world logistics data with interpretable predictions.

## What This Does

FlowSight predicts whether shipments will be delayed, estimates delay duration, identifies probable delay reasons, and explains predictions in plain language. The system processes shipment metadata (vendor info, routes, historical patterns) and returns actionable predictions through a FastAPI backend and Streamlit dashboard.

## Architecture

```
├── src/
│   ├── data/           # Data ingestion and processing (DuckDB-based)
│   ├── enrichment/     # Feature engineering pipeline
│   ├── models/         # Model training and inference
│   ├── api/            # FastAPI endpoints
│   ├── dashboard/      # Streamlit UI
│   └── explainability/ # LLM-powered prediction explanations
├── config/             # Configuration management
├── models/             # Trained model artifacts
└── data/               # Raw and processed datasets
```

## Key Features

- **Multi-model predictions**: Ensemble of CatBoost classifiers for delay detection, duration estimation, and reason classification
- **Real-time inference**: Sub-second predictions via FastAPI with async handling
- **Explainability**: LLM integration to convert model outputs into business-friendly explanations
- **Feature store**: Centralized feature management with DuckDB for efficient querying
- **Enrichment layer**: Automated feature engineering from raw shipment data

## Tech Stack

- **ML**: CatBoost, scikit-learn, pandas
- **Backend**: FastAPI, Pydantic
- **Data**: DuckDB (feature store), parquet files
- **Frontend**: Streamlit
- **Explainability**: LLM-based (OpenAI/Anthropic compatible)

## Setup

1. Clone and install dependencies:
```bash
git clone https://github.com/pranaya-mathur/flowsight-ai-.git
cd flowsight-ai-
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Add your API keys and database paths
```

3. Run the API:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Launch the dashboard:
```bash
streamlit run src/dashboard/app.py
```

## API Usage

### Predict Delay
```python
import requests

payload = {
    "vendor_id": "V12345",
    "route_id": "R789",
    "shipment_date": "2026-01-15",
    "quantity": 1500,
    "distance_km": 450
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

**Response:**
```json
{
  "will_delay": true,
  "delay_probability": 0.78,
  "estimated_delay_days": 3.2,
  "delay_reason": "weather_disruption",
  "explanation": "High probability of delay due to vendor's recent performance issues and adverse weather conditions on route."
}
```

## Model Training

Models are trained using historical shipment data with features like:
- Vendor reliability scores
- Route complexity metrics
- Seasonal patterns
- Historical delay rates
- Distance and logistics parameters

Training scripts and notebooks are in `src/models/`. The current ensemble achieves ~85% accuracy on delay classification with mean absolute error of 1.2 days on delay duration estimates.

## Project Structure

- `diagnostic_*.py`: Schema inspection utilities
- `inspect_*.py`: Feature store and training data validation
- `test_*.py`: End-to-end testing for all components
- `src/exceptions.py`: Custom exception handling
- `src/logging_utils.py`: Centralized logging configuration

## Development Notes

The repo uses CatBoost for traditional ML due to superior performance vs neural approaches (tested transformers and multi-task NNs with worse results). Model files are excluded from version control due to size - they're generated during training.

## Roadmap

- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Model versioning with DVC
- [ ] A/B testing framework
- [ ] Performance monitoring dashboard

## License

MIT

---

Built for production supply chain environments. PRs welcome.
