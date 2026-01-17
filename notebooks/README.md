# Notebooks

Experimental notebooks for data exploration and model development.

## Contents (WIP)

### Data Exploration
- [ ] `01_data_exploration.ipynb` - Initial data analysis, distribution plots
- [ ] `02_feature_analysis.ipynb` - Feature importance, correlations
- [ ] `03_vendor_route_patterns.ipynb` - Vendor/route performance deep dive

### Model Development
- [ ] `04_baseline_models.ipynb` - Simple logistic regression, decision trees
- [ ] `05_ensemble_experiments.ipynb` - Hyperparameter tuning for ensemble
- [ ] `06_transformer_attempts.ipynb` - Failed transformer experiments (kept for reference)

### Analysis
- [ ] `07_error_analysis.ipynb` - Where does the model fail?
- [ ] `08_enrichment_impact.ipynb` - A/B test enrichment layer
- [ ] `09_llm_explanation_quality.ipynb` - Evaluate explanation usefulness

## TODO

Need to clean up and commit actual notebooks. Currently they're messy prototypes on local machine.

## Usage

```bash
jupyter notebook
# or
jupyter lab
```

## Notes

- Notebooks use relative imports from `src/`
- Some require trained models in `models/`
- Data files should be in `data/raw/`
