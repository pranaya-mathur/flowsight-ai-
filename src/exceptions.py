class FlowSightError(Exception):
    """Base exception for FlowSight."""


class DataLoadError(FlowSightError):
    """Raised when raw data cannot be loaded or validated."""


class FeatureStoreError(FlowSightError):
    """Raised when feature store operations fail."""


class ModelTrainingError(FlowSightError):
    """Raised when model training fails."""


class ModelInferenceError(FlowSightError):
    """Raised when prediction/inference fails."""


class EnrichmentError(FlowSightError):
    """Raised when vendor/route enrichment fails."""


class ExplainabilityError(FlowSightError):
    """Raised for SHAP or LLM explanation failures."""
