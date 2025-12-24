from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import settings
from src.data.features import FeatureBuilder
from src.exceptions import ModelTrainingError
from src.logging_utils import get_logger

logger = get_logger(__name__)


NUMERIC_COLS: List[str] = [
    "quantity",
    "weight_kg",
    "value_inr",
    "risk_score",
    "vendor_avg_delay_days",
    "vendor_on_time_rate",
    "route_distance_km",
    "route_vendor_reliability_sim",
    "route_delay_probability_sim",
    "value_per_kg",
]

CATEGORICAL_COLS: List[str] = [
    "origin_city",
    "destination_city",
    "origin_state",
    "destination_state",
    "product_category",
    "supplier_name",
    "carrier_name",
    "truck_type",
    "month",
    "route_pair",
]


def _build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    feature_cols = num_cols + cat_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, feature_cols


def train_delay_regressor() -> None:
    try:
        fb = FeatureBuilder()
        df = fb.build_training_frame()
    except Exception as exc:
        logger.exception("Failed to build training data for delay regressor")
        raise ModelTrainingError("Could not build training data for delay regressor") from exc

    # Focus on shipments that are actually delayed (delay_days > 0)
    df_reg = df.copy()
    df_reg["delay_days"] = df_reg["delay_days"].fillna(0).astype(float)
    df_reg = df_reg[df_reg["delay_days"] > 0].reset_index(drop=True)

    if df_reg.empty:
        raise ModelTrainingError("No delayed shipments found for regression training")

    logger.info("Delay regressor: using %d delayed rows out of %d", df_reg.shape[0], df.shape[0])

    preprocessor, feature_cols = _build_preprocessor(df_reg)

    X = df_reg[feature_cols].copy()
    y = df_reg["delay_days"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    logger.info(
        "Delay regressor: train=%d, val=%d, features=%d",
        X_train.shape[0],
        X_val.shape[0],
        len(feature_cols),
    )

    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        logger.exception("Training delay regressor failed")
        raise ModelTrainingError("Training delay regressor failed") from exc

    # Validation
    y_pred = pipeline.predict(X_val)
    # Clamp negative predictions
    y_pred = np.clip(y_pred, 0.0, None)

    mae = mean_absolute_error(y_val, y_pred)
    logger.info("Delay regressor validation MAE=%.4f days", mae)

    # Persist artifacts
    try:
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = settings.MODELS_DIR / "delay_days_regressor.pkl"
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_cols": feature_cols,
            },
            out_path,
        )
        logger.info("Saved delay_days regressor to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save delay regressor")
        raise ModelTrainingError("Failed to save delay regressor artifacts") from exc


if __name__ == "__main__":
    train_delay_regressor()
