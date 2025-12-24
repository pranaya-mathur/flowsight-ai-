from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import settings
from src.data.features import FeatureBuilder, DEFAULT_FEATURE_CONFIG
from src.exceptions import ModelTrainingError
from src.logging_utils import get_logger

logger = get_logger(__name__)


def _train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["will_delay"].astype(int),
    )
    return train_df, val_df


def _build_preprocessor(feature_df: pd.DataFrame) -> Tuple[ColumnTransformer, list]:
    cat_cols = [c for c in DEFAULT_FEATURE_CONFIG.categorical_cols if c in feature_df.columns]
    num_cols = [c for c in DEFAULT_FEATURE_CONFIG.numeric_cols if c in feature_df.columns]
    extra_cols = [c for c in ["value_per_kg", "route_pair"] if c in feature_df.columns]

    # route_pair is categorical
    if "route_pair" in extra_cols and "route_pair" not in cat_cols:
        cat_cols.append("route_pair")
        extra_cols.remove("route_pair")

    feature_cols = num_cols + cat_cols + extra_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols + extra_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, feature_cols


def train_ensemble() -> None:
    try:
        fb = FeatureBuilder()
        df = fb.build_training_frame()
    except Exception as exc:
        logger.exception("Failed to build training data for ensemble")
        raise ModelTrainingError("Could not build training data") from exc

    logger.info("Full training frame: %d rows, %d columns", *df.shape)

    train_df, val_df = _train_val_split(df)

    preprocessor, feature_cols = _build_preprocessor(train_df)

    X_train = train_df[feature_cols].copy()
    y_train = train_df["will_delay"].astype(int).values
    X_val = val_df[feature_cols].copy()
    y_val = val_df["will_delay"].astype(int).values

    logger.info(
        "Training data: train=%d, val=%d, features=%d",
        X_train.shape[0],
        X_val.shape[0],
        len(feature_cols),
    )

    # Fit preprocessing on training data only
    try:
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
    except Exception as exc:
        logger.exception("Preprocessing failed")
        raise ModelTrainingError("Feature preprocessing failed") from exc

    logger.info(
        "After preprocessing: train=%s, val=%s",
        X_train_proc.shape,
        X_val_proc.shape,
    )

    try:
        # Models all receive the same numeric matrices
        cat_model = CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            loss_function="Logloss",
            verbose=False,
            random_state=42,
        )
        lgbm_model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
        )
        xgb_model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
        )

        logger.info("Fitting CatBoost...")
        cat_model.fit(X_train_proc, y_train)

        logger.info("Fitting LightGBM...")
        lgbm_model.fit(X_train_proc, y_train)

        logger.info("Fitting XGBoost...")
        xgb_model.fit(X_train_proc, y_train)
    except Exception as exc:
        logger.exception("Model training failed")
        raise ModelTrainingError("Training one or more ensemble models failed") from exc

    # Validation predictions
    try:
        cat_probs = cat_model.predict_proba(X_val_proc)[:, 1]
        lgbm_probs = lgbm_model.predict_proba(X_val_proc)[:, 1]
        xgb_probs = xgb_model.predict_proba(X_val_proc)[:, 1]

        ensemble_probs = (cat_probs + lgbm_probs + xgb_probs) / 3.0
        ensemble_pred = (ensemble_probs >= 0.5).astype(int)

        acc = accuracy_score(y_val, ensemble_pred)
        auc = roc_auc_score(y_val, ensemble_probs)
        logger.info("Ensemble validation accuracy=%.4f, AUC=%.4f", acc, auc)
    except Exception as exc:
        logger.exception("Validation/evaluation failed")
        raise ModelTrainingError("Evaluation of ensemble models failed") from exc

    # Persist everything needed for inference
    try:
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = settings.MODELS_DIR / "ensemble_binary_delay.pkl"
        joblib.dump(
            {
                "preprocessor": preprocessor,
                "catboost": cat_model,
                "lightgbm": lgbm_model,
                "xgboost": xgb_model,
                "feature_cols": feature_cols,
            },
            out_path,
        )
        logger.info("Saved ensemble model and preprocessor to %s", out_path)
    except Exception as exc:
        logger.exception("Failed to save ensemble artifacts")
        raise ModelTrainingError("Failed to save ensemble models") from exc


if __name__ == "__main__":
    train_ensemble()
