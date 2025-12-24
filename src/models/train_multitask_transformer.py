"""
Train a multi-task Transformer model for FlowSight.

Tasks (shared backbone + 3 heads):
- Binary: will_delay
- Regression: delay_days
- Multi-class: delay_reason (including "None")
"""

import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.features import FeatureBuilder, DEFAULT_FEATURE_CONFIG  # type: ignore
from src.exceptions import DataLoadError, ModelTrainingError  # type: ignore
from src.logging_utils import get_logger  # type: ignore

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y_binary: np.ndarray,
        y_reg: np.ndarray,
        y_class: np.ndarray,
    ) -> None:
        self.X_num = torch.from_numpy(X_num.astype(np.float32))
        self.X_cat = torch.from_numpy(X_cat.astype(np.int64))
        self.y_binary = torch.from_numpy(y_binary.astype(np.float32))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32))
        self.y_class = torch.from_numpy(y_class.astype(np.int64))

    def __len__(self) -> int:
        return self.X_num.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y_binary[idx],
            self.y_reg[idx],
            self.y_class[idx],
        )


# -------------------------------------------------------------------
# Model: Transformer backbone + 3 heads
# -------------------------------------------------------------------


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        num_num_features: int,
        cat_cardinalities: List[int],
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.num_proj = nn.Linear(num_num_features, d_model)

        self.cat_embeddings = nn.ModuleList()
        self.cat_projections = nn.ModuleList()
        for card in cat_cardinalities:
            emb_dim = min(32, max(8, card // 2))
            emb = nn.Embedding(num_embeddings=card, embedding_dim=emb_dim)
            proj = nn.Linear(emb_dim, d_model)
            self.cat_embeddings.append(emb)
            self.cat_projections.append(proj)

        seq_len = 1 + len(cat_cardinalities)
        self.positional = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num_token = self.num_proj(x_num).unsqueeze(1)  # [B, 1, d_model]

        cat_tokens = []
        for i, (emb, proj) in enumerate(zip(self.cat_embeddings, self.cat_projections)):
            cat_emb = emb(x_cat[:, i])               # [B, emb_dim]
            cat_emb = proj(cat_emb).unsqueeze(1)     # [B, 1, d_model]
            cat_tokens.append(cat_emb)

        x = torch.cat([num_token] + cat_tokens, dim=1)  # [B, 1+F_cat, d_model]
        x = x + self.positional[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.layer_norm(x)
        x = self.dropout(x.mean(dim=1))  # [B, d_model]
        return x


class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        num_num_features: int,
        cat_cardinalities: List[int],
        num_delay_reason_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = TransformerBackbone(num_num_features, cat_cardinalities)
        d_model = self.backbone.d_model

        self.head_binary = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self.head_regression = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.head_multiclass = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_delay_reason_classes),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        shared = self.backbone(x_num, x_cat)
        bin_logits = self.head_binary(shared).squeeze(-1)
        reg_out = self.head_regression(shared).squeeze(-1)
        cls_logits = self.head_multiclass(shared)
        return bin_logits, reg_out, cls_logits


# -------------------------------------------------------------------
# Preprocessing using FeatureBuilder
# -------------------------------------------------------------------


def build_training_frame() -> pd.DataFrame:
    try:
        fb = FeatureBuilder()
        df = fb.build_training_frame()
        return df
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build training frame.")
        raise DataLoadError("Failed to build training frame.") from exc


def preprocess(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict, List[str], List[str]]:
    """
    - Uses DEFAULT_FEATURE_CONFIG for base numeric/categorical columns.
    - Adds safe log features.
    - Drops all-NaN enrichment columns.
    """
    try:
        logger.info("Starting preprocessing...")

        # Targets
        y_binary = df["will_delay"].astype("int32").to_numpy()
        y_reg = df["delay_days"].astype("float32").to_numpy()
        y_reg = np.clip(y_reg, 0.0, 12.0)  # safety; your data already 0–12. [conversation_history:0]

        delay_reason_le = LabelEncoder()
        df["delay_reason_filled"] = df["delay_reason"].fillna("None")
        y_class = delay_reason_le.fit_transform(df["delay_reason_filled"]).astype("int64")

        # Numeric features
        numeric_cols = DEFAULT_FEATURE_CONFIG.numeric_cols.copy()
        # All current enrichments are NaN; keep them but they will be dropped by variance/NaN check. [conversation_history:0]

        df["log_value_inr"] = np.log1p(df["value_inr"])
        df["log_weight_kg"] = np.log1p(df["weight_kg"])
        df["log_quantity"] = np.log1p(df["quantity"])

        extra_num = ["log_value_inr", "log_weight_kg", "log_quantity"]
        num_features = [c for c in numeric_cols + extra_num if c in df.columns]

        # Drop columns that are entirely NaN
        num_features = [
            c for c in num_features if not df[c].isna().all()
        ]

        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[num_features].astype("float32"))

        # Categorical features
        cat_features = [c for c in DEFAULT_FEATURE_CONFIG.categorical_cols if c in df.columns]

        cat_encoders: Dict[str, LabelEncoder] = {}
        X_cat = np.zeros((len(df), len(cat_features)), dtype="int64")
        for idx, col in enumerate(cat_features):
            le = LabelEncoder()
            X_cat[:, idx] = le.fit_transform(df[col].astype(str))
            cat_encoders[col] = le

        logger.info(
            "Preprocessing complete. Num features: %d, Cat features: %d, delay_reason classes: %d",
            len(num_features),
            len(cat_features),
            len(delay_reason_le.classes_),
        )

        artifacts = {
            "scaler": scaler,
            "delay_reason_le": delay_reason_le,
        }

        return X_num, X_cat, y_binary, y_reg, y_class, cat_encoders, artifacts, num_features, cat_features
    except Exception as exc:  # noqa: BLE001
        logger.exception("Preprocessing failed.")
        raise DataLoadError("Failed to preprocess training data.") from exc


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_bin, all_bin_true = [], []
    all_reg, all_reg_true = [], []
    all_cls, all_cls_true = [], []

    with torch.no_grad():
        for X_num, X_cat, y_bin, y_reg, y_cls in loader:
            X_num = X_num.to(device)
            X_cat = X_cat.to(device)
            y_bin = y_bin.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            bin_logits, reg_out, cls_logits = model(X_num, X_cat)
            all_bin.append(torch.sigmoid(bin_logits).cpu().numpy())
            all_bin_true.append(y_bin.cpu().numpy())
            all_reg.append(reg_out.cpu().numpy())
            all_reg_true.append(y_reg.cpu().numpy())
            all_cls.append(cls_logits.cpu().numpy())
            all_cls_true.append(y_cls.cpu().numpy())

    bin_pred = np.concatenate(all_bin)
    bin_true = np.concatenate(all_bin_true)
    reg_pred = np.concatenate(all_reg)
    reg_true = np.concatenate(all_reg_true)
    cls_logits = np.concatenate(all_cls)
    cls_true = np.concatenate(all_cls_true)

    bin_auc = roc_auc_score(bin_true, bin_pred)
    reg_mae = mean_absolute_error(reg_true, reg_pred)
    cls_pred = cls_logits.argmax(axis=1)
    macro_f1 = f1_score(cls_true, cls_pred, average="macro")

    return {"binary_auc": bin_auc, "reg_mae": reg_mae, "macro_f1": macro_f1}


# -------------------------------------------------------------------
# Training entrypoint
# -------------------------------------------------------------------


def train_multitask_transformer(
    output_dir: str = "models",
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 1) Build training frame with existing pipeline
        df = build_training_frame()

        # 2) Preprocess
        (
            X_num,
            X_cat,
            y_binary,
            y_reg,
            y_class,
            cat_encoders,
            artifacts,
            num_features,
            cat_features,
        ) = preprocess(df)

        # 3) Split
        (
            X_num_tr,
            X_num_val,
            X_cat_tr,
            X_cat_val,
            y_bin_tr,
            y_bin_val,
            y_reg_tr,
            y_reg_val,
            y_cls_tr,
            y_cls_val,
        ) = train_test_split(
            X_num,
            X_cat,
            y_binary,
            y_reg,
            y_class,
            test_size=0.2,
            random_state=seed,
            stratify=y_binary,
        )

        train_ds = MultiTaskDataset(X_num_tr, X_cat_tr, y_bin_tr, y_reg_tr, y_cls_tr)
        val_ds = MultiTaskDataset(X_num_val, X_cat_val, y_bin_val, y_reg_val, y_cls_val)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=256, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=256, shuffle=False, num_workers=0
        )

        cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_features]
        num_delay_reason_classes = len(artifacts["delay_reason_le"].classes_)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        model = MultiTaskTransformer(
            num_num_features=X_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            num_delay_reason_classes=num_delay_reason_classes,
        ).to(device)

        # Losses
        criterion_bin = nn.BCEWithLogitsLoss()
        criterion_reg = nn.HuberLoss(delta=1.0)

        class_counts = np.bincount(y_class)
        class_counts = np.where(class_counts == 0, 1, class_counts)
        inv_freq = 1.0 / class_counts
        class_weights = (inv_freq / inv_freq.mean()).astype(np.float32)
        if not np.isfinite(class_weights).all():
            logger.warning("Non-finite class weights detected; using uniform weights.")
            class_weights = np.ones_like(class_weights, dtype=np.float32)

        criterion_cls = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).to(device)
        )

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        best_macro_f1 = -1.0
        best_state = None
        max_epochs = 40

        logger.info("Starting multi-task training for up to %d epochs...", max_epochs)

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for X_num_b, X_cat_b, y_bin_b, y_reg_b, y_cls_b in train_loader:
                X_num_b = X_num_b.to(device)
                X_cat_b = X_cat_b.to(device)
                y_bin_b = y_bin_b.to(device)
                y_reg_b = y_reg_b.to(device)
                y_cls_b = y_cls_b.to(device)

                optimizer.zero_grad()
                bin_logits, reg_out, cls_logits = model(X_num_b, X_cat_b)

                loss_bin = criterion_bin(bin_logits, y_bin_b)
                loss_reg = criterion_reg(reg_out, y_reg_b)
                loss_cls = criterion_cls(cls_logits, y_cls_b)

                loss = loss_bin + loss_reg + 0.7 * loss_cls

                if not torch.isfinite(loss):
                    logger.error(
                        "Non-finite loss detected (bin=%.4f, reg=%.4f, cls=%.4f).",
                        loss_bin.item(),
                        loss_reg.item(),
                        loss_cls.item(),
                    )
                    raise ModelTrainingError("Non-finite loss during training.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            metrics = evaluate(model, val_loader, device)
            scheduler.step(metrics["macro_f1"])

            logger.info(
                "Epoch %02d | loss=%.4f | val_auc=%.4f | val_mae=%.4f | val_macro_f1=%.4f",
                epoch,
                epoch_loss / max(1, len(train_loader)),
                metrics["binary_auc"],
                metrics["reg_mae"],
                metrics["macro_f1"],
            )

            if np.isfinite(metrics["macro_f1"]) and metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_state = model.state_dict()

        if best_state is None:
            logger.warning(
                "No valid best state from validation; falling back to final model weights."
            )
            best_state = model.state_dict()

        model.load_state_dict(best_state)

        # Save model + preprocess artifacts
        model_path = os.path.join(output_dir, "multitask_transformer.pt")
        torch.save(model.state_dict(), model_path)

        joblib.dump(artifacts["scaler"], os.path.join(output_dir, "mt_scaler_num.pkl"))
        joblib.dump(cat_encoders, os.path.join(output_dir, "mt_cat_encoders.pkl"))
        joblib.dump(
            artifacts["delay_reason_le"],
            os.path.join(output_dir, "mt_delay_reason_le.pkl"),
        )
        joblib.dump(num_features, os.path.join(output_dir, "mt_num_features.pkl"))
        joblib.dump(cat_features, os.path.join(output_dir, "mt_cat_features.pkl"))

        logger.info(
            "Training complete. Best validation macro-F1: %.4f. Model saved to %s",
            best_macro_f1,
            model_path,
        )

    except (DataLoadError, ModelTrainingError):
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during multi-task Transformer training.")
        raise ModelTrainingError(
            "Unexpected failure during multi-task Transformer training."
        ) from exc


if __name__ == "__main__":
    train_multitask_transformer()
