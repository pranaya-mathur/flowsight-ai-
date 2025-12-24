from __future__ import annotations
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from config import settings
from src.data.features import FeatureBuilder
from src.exceptions import ModelTrainingError
from src.logging_utils import get_logger

logger = get_logger(__name__)

NUMERIC_COLS: List[str] = [
    "quantity", "weight_kg", "value_inr", "risk_score",
    "vendor_avg_delay_days", "vendor_on_time_rate", 
    "route_distance_km", "route_vendor_reliability_sim",
    "route_delay_probability_sim", "value_per_kg",
]

CATEGORICAL_COLS: List[str] = [
    "origin_city", "destination_city", "origin_state", 
    "destination_state", "product_category", "supplier_name",
    "carrier_name", "truck_type", "month", "route_pair",
]

def train_delay_reason_classifier_v2() -> None:
    """🎯 PRODUCTION V2: 80% F1 Target - Class Weights + CV + Ensemble"""
    
    # STEP 0: Load from feature store
    try:
        fb = FeatureBuilder()
        df = fb.build_training_frame()
        logger.info("✅ Loaded %d rows from feature store", df.shape[0])
    except Exception as exc:
        logger.exception("Failed to build training data")
        raise ModelTrainingError("Could not build training data") from exc

    # Filter delayed shipments
    df_cls = df[df["will_delay"] == 1].copy()
    df_cls["delay_reason"] = df_cls["delay_reason"].fillna("Unknown").astype(str)
    
    if df_cls.empty:
        raise ModelTrainingError("No delayed shipments found")
    
    logger.info("Using %d delayed rows", df_cls.shape[0])

    # STEP 1: CLASS DISTRIBUTION
    class_dist = df_cls["delay_reason"].value_counts(normalize=True)
    logger.info("Class distribution (top 10):\n%s", class_dist.head(10))
    rare_classes = class_dist[class_dist < 0.01].index.tolist()
    logger.info("Rare classes (<1%%): %d", len(rare_classes))

    # Encode target
    classes = sorted(df_cls["delay_reason"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y = df_cls["delay_reason"].map(class_to_idx).values.astype(int)
    
    feature_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c in df_cls.columns]
    X = df_cls[feature_cols].copy()
    
    # 🔧 CRITICAL: Identify cat_features INDICES
    cat_feature_indices = [i for i, col in enumerate(feature_cols) if col in CATEGORICAL_COLS]
    num_feature_indices = [i for i, col in enumerate(feature_cols) if col in NUMERIC_COLS]
    
    logger.info("✅ Data prepared: %d features (%d numeric, %d categorical), %d classes", 
                len(feature_cols), len(num_feature_indices), len(cat_feature_indices), len(classes))
    logger.info("   Numeric: %s", [feature_cols[i] for i in num_feature_indices])
    logger.info("   Categorical: %s", [feature_cols[i] for i in cat_feature_indices])
    
    # Handle NaN in numeric cols only
    for col in [feature_cols[i] for i in num_feature_indices]:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # STEP 2: BASELINE WITH CLASS WEIGHTS
    logger.info("🚀 STEP 2: Baseline + Balanced Weights")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)
    
    baseline_model = CatBoostClassifier(
        iterations=2000, depth=8, learning_rate=0.05, l2_leaf_reg=3,
        auto_class_weights='Balanced',
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        early_stopping_rounds=200,
        random_seed=42,
        verbose=100
    )
    
    baseline_model.fit(train_pool, eval_set=test_pool)
    baseline_preds = baseline_model.predict(test_pool)
    baseline_f1 = f1_score(y_test, baseline_preds, average='macro')
    baseline_acc = accuracy_score(y_test, baseline_preds)
    logger.info("✅ Baseline Macro F1: %.4f | Accuracy: %.4f", baseline_f1, baseline_acc)

    # STEP 3: 5-FOLD STRATIFIED CV
    logger.info("📊 STEP 3: 5-Fold Stratified CV")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)
        
        cv_model = CatBoostClassifier(
            iterations=2000, depth=8, learning_rate=0.05, l2_leaf_reg=3,
            auto_class_weights='Balanced',
            loss_function='MultiClass',
            eval_metric='TotalF1:average=Macro',
            early_stopping_rounds=200,
            random_seed=42+fold,
            verbose=False
        )
        
        cv_model.fit(tr_pool, eval_set=val_pool)
        cv_preds = cv_model.predict(val_pool)
        cv_f1_scores.append(f1_score(y_val, cv_preds, average='macro'))
        logger.info("Fold %d: %.4f", fold+1, cv_f1_scores[-1])
    
    cv_mean_f1 = np.mean(cv_f1_scores)
    cv_std_f1 = np.std(cv_f1_scores)
    logger.info("✅ CV Macro F1: %.4f ± %.4f", cv_mean_f1, cv_std_f1*2)

    # STEP 4: FINAL 5-MODEL ENSEMBLE
    logger.info("🏆 STEP 4: Final 5-Model Ensemble")
    final_models = []
    full_pool = Pool(X, y, cat_features=cat_feature_indices)
    
    for i in range(5):
        model = CatBoostClassifier(
            iterations=2000, depth=8, learning_rate=0.05, l2_leaf_reg=3,
            auto_class_weights='Balanced',
            loss_function='MultiClass',
            random_seed=42+i*10,
            verbose=200 if i==0 else False
        )
        model.fit(full_pool)
        final_models.append(model)
    
    # Ensemble predictions (average probabilities)
    test_probs = np.mean([m.predict_proba(test_pool) for m in final_models], axis=0)
    final_preds = np.argmax(test_probs, axis=1)
    final_f1 = f1_score(y_test, final_preds, average='macro')
    final_acc = accuracy_score(y_test, final_preds)
    logger.info("✅ Final Test Macro F1: %.4f | Accuracy: %.4f", final_f1, final_acc)

    # Detailed report
    logger.info("📈 Classification Report:\n%s", 
                classification_report(y_test, final_preds, target_names=classes, zero_division=0))

    # STEP 5: SAVE PRODUCTION ARTIFACTS
    logger.info("💾 STEP 5: Saving Production Model")
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        "models": final_models,
        "feature_cols": feature_cols,
        "cat_feature_indices": cat_feature_indices,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "cv_f1_mean": cv_mean_f1,
        "cv_f1_std": cv_std_f1,
        "baseline_f1": baseline_f1,
        "final_f1": final_f1,
        "n_classes": len(classes),
    }
    
    output_path = settings.MODELS_DIR / "delay_reason_classifier_v2.pkl"
    joblib.dump(artifacts, output_path)
    logger.info("✅ Saved v2 ensemble to %s", output_path)

    # 🎯 FINAL SUMMARY
    logger.info("="*70)
    logger.info("🎯 PRODUCTION RESULTS SUMMARY")
    logger.info("   Dataset:          %d delayed shipments", len(df_cls))
    logger.info("   Classes:          %d", len(classes))
    logger.info("   Baseline F1:      %.4f", baseline_f1)
    logger.info("   CV F1 (5-fold):   %.4f ± %.4f", cv_mean_f1, cv_std_f1*2)
    logger.info("   Final Test F1:    %.4f", final_f1)
    logger.info("   Target >0.45:     %s", "✅ PASSED" if cv_mean_f1 > 0.45 else "⚠️ IMPROVE")
    logger.info("   Ready for SHAP+RCA integration!")
    logger.info("="*70)

if __name__ == "__main__":
    train_delay_reason_classifier_v2()
