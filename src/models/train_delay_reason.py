from __future__ import annotations

from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
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


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

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
    return preprocessor


def train_delay_reason_classifier() -> None:
    try:
        fb = FeatureBuilder()
        df = fb.build_training_frame()
    except Exception as exc:
        logger.exception("Failed to build training data for delay_reason classifier")
        raise ModelTrainingError("Could not build training data for delay_reason") from exc

    # Use only delayed shipments with a defined reason
    df_cls = df.copy()
    df_cls = df_cls[df_cls["will_delay"] == 1]
    df_cls["delay_reason"] = df_cls["delay_reason"].fillna("Unknown").astype(str)

    if df_cls.empty:
        raise ModelTrainingError("No delayed shipments with reasons found for classification")

    logger.info(
        "Delay_reason classifier: using %d delayed rows out of %d",
        df_cls.shape[0],
        df.shape[0],
    )

    # Encode target
    classes = sorted(df_cls["delay_reason"].unique())
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
    idx_to_class: Dict[int, str] = {i: c for c, i in class_to_idx.items()}
    y = df_cls["delay_reason"].map(class_to_idx).values.astype(int)

    preprocessor = _build_preprocessor(df_cls)

    feature_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df_cls.columns]
    X = df_cls[feature_cols].copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(
        "Delay_reason classifier: train=%d, val=%d, features=%d, classes=%d",
        X_train.shape[0],
        X_val.shape[0],
        len(feature_cols),
        len(classes),
    )

    # CatBoost handling numeric matrix after preprocessing for consistency
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        verbose=False,
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
        logger.exception("Training delay_reason classifier failed")
        raise ModelTrainingError("Training delay_reason classifier failed") from exc

    # Validation
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info("Delay_reason validation accuracy=%.4f", acc)

    # Optional detailed report in logs
    rep = classification_report(
        y_val,
        y_pred,
        target_names=classes,
        zero_division=0,
    )
    logger.info("Delay_reason classification report:\n%s", rep)

    # Persist artifacts
    try:
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = settings.MODELS_DIR / "delay_reason_classifier.pkl"
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_cols": feature_cols,
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
            },
            out_path,
        )
        logger.info("Saved delay_reason classifier to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save delay_reason classifier")
        raise ModelTrainingError("Failed to save delay_reason classifier artifacts") from exc


if __name__ == "__main__":
    train_delay_reason_classifier()
