class MLBPredictionError(Exception):
    """Base exception for this project."""


class RateLimitError(MLBPredictionError):
    """Raised when an API returns 429 Too Many Requests."""


class DataCollectionError(MLBPredictionError):
    """Raised when data collection fails after retries."""


class FeatureEngineeringError(MLBPredictionError):
    """Raised when feature computation fails."""


class ModelNotFoundError(MLBPredictionError):
    """Raised when a trained model file is missing."""
