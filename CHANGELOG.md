# Changelog

All notable changes to FlowSight AI will be documented here.

## [Unreleased]

### Todo
- [ ] Add model versioning system (DVC or MLflow)
- [ ] Implement caching for DuckDB vendor/route lookups
- [ ] Dashboard performance optimization - takes 3-4s for predictions
- [ ] Add batch prediction endpoint
- [ ] Write proper unit tests (current coverage ~40%)
- [ ] Add monitoring/logging for production deployments

### Known Issues
- Dashboard occasionally throws NaN errors with invalid vendor IDs
- LLM explanations timeout on slow networks (30s limit)
- Route validation confidence calculation needs refinement

## [0.2.0] - 2025-12-24

### Added
- LLM-powered explanations using Groq/LLaMA 3.3
- Interactive Streamlit dashboard with visualizations
- Waterfall charts showing enrichment impact
- Health check endpoint for monitoring
- Request/response schemas with Pydantic validation

### Changed
- Switched from single model to ensemble approach (improved accuracy from 78% to 85%)
- Moved from in-memory lookups to DuckDB feature store
- Refactored inference pipeline to support optional enrichment

### Fixed
- Numpy type serialization issues in API responses
- Vendor adjustment edge case when on_time_rate is exactly 0.5

## [0.1.0] - 2025-12-23

### Added
- Initial ensemble model (CatBoost + XGBoost + LightGBM)
- Binary delay classifier
- Delay days regressor  
- Delay reason classifier
- Vendor enrichment layer
- Route validation system
- DuckDB feature store implementation
- FastAPI backend with /predict endpoint

### Tried and Abandoned
- Multi-task transformer model - terrible results (AUC 0.61 vs 0.89 for ensemble)
- Neural network approach - overfit badly on small dataset
- Single CatBoost model - ensemble performed 3-4% better

### Notes
- Initial data exploration showed heavy class imbalance (67% delays)
- Vendor on-time rates range from 23% to 89%
- Route Mumbai→Delhi has highest volume (847 shipments)
