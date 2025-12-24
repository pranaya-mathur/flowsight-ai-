"""
Train a multi-task Transformer model for FlowSight.

Tasks (shared backbone + 3 heads):
- Binary: will_delay
- Regression: delay_days
- Multi-class: delay_reason (including "None")

Expected to run after feature building has created the DuckDB training table.
"""

import os
import sys
from typing import Dict, List, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

# Adjust path so this works when called as a module or script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.logging_utils import get_logger  # type: ignore
from src.exceptions import DataLoadError, ModelTrainingError  # type: ignore

logger = get_logger(__name__)


# -------------------------------------------------------------------
# PyTorch Dataset
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
# Model definition: Transformer backbone + 3 heads
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

        # Numeric projection: [B, F_num] -> [B, 1, d_model]
        self.num_proj = nn.Linear(num_num_features, d_model)

        # Categorical embeddings: one token per categorical feature
        self.cat_embeddings = nn.ModuleList()
        self.cat_projections = nn.ModuleList()
        for card in cat_cardinalities:
            emb_dim = min(32, max(8, card // 2))
            emb = nn.Embedding(num_embeddings=card, embedding_dim=emb_dim)
            proj = nn.Linear(emb_dim, d_model)
            self.cat_embeddings.append(emb)
            self.cat_projections.append(proj)

        # Positional encoding for [num_token + cat_tokens]
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
        # x_num: [B, F_num]
        # x_cat: [B, F_cat]
        num_token = self.num_proj(x_num).unsqueeze(1)  # [B, 1, d_model]

        cat_tokens = []
        for i, (emb, proj) in enumerate(zip(self.cat_embeddings, self.cat_projections)):
            cat_emb = emb(x_cat[:, i])  # [B, emb_dim]
            cat_emb = proj(cat_emb).unsqueeze(1)  # [B, 1, d_model]
            cat_tokens.append(cat_emb)

        x = torch.cat([num_token] + cat_tokens, dim=1)  # [B, 1+F_cat, d_model]
        x = x + self.positional[:, : x.size(1), :]
        x = self.encoder(x)
        x = self.layer_norm(x)
        x = self.dropout(x.mean(dim=1))  # global average pooling -> [B, d_model]
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

        # Binary head: will_delay
        self.head_binary = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # Regression head: delay_days
        self.head_regression = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Multi-class head: delay_reason (including "None")
        self.head_multiclass = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_delay_reason_classes),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        shared = self.backbone(x_num, x_cat)
        binary_logits = self.head_binary(shared).squeeze(-1)
        reg_output = self.head_regression(shared).squeeze(-1)
        class_logits = self.head_multiclass(shared)
        return binary_logits, reg_output, class_logits


# -------------------------------------------------------------------
# Data preparation from DuckDB
# -------------------------------------------------------------------


def load_training_frame(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    try:
        logger.info("Loading training frame from DuckDB...")
        # Use the columns as shown in inspect_training_frame.py
        query = """
        SELECT
            quantity,
            weight_kg,
            value_inr,
            gst_rate,
            risk_score,
            origin_city,
            destination_city,
            origin_state,
            destination_state,
            product_category,
            supplier_name,
            carrier_name,
            truck_type,
            month,
            will_delay,
            delay_days,
            delay_reason
        FROM training_frame
        """
        df = conn.execute(query).df()
        if df.empty:
            raise DataLoadError("Training frame query returned no rows.")
        logger.info("Loaded training frame with %d rows.", len(df))
        return df
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load training frame.")
        raise DataLoadError("Failed to load training frame from DuckDB.") from exc


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict, Dict, List[str], List[str]]:
    """
    Prepare numeric & categorical matrices and targets.

    - Log transforms for skewed numerics
    - Standard scaling for numerics
    - LabelEncoding for categoricals and delay_reason
    """
    try:
        logger.info("Starting preprocessing...")

        # Targets
        y_binary = df["will_delay"].astype("int32").to_numpy()
        y_reg = df["delay_days"].astype("float32").to_numpy()

        # Multiclass target: delay_reason (None + 10 real reasons)
        delay_reason_le = LabelEncoder()
        df["delay_reason_filled"] = df["delay_reason"].fillna("None")
        y_class = delay_reason_le.fit_transform(df["delay_reason_filled"]).astype("int64")

        # Numeric features: raw + engineered
        df["log_value_inr"] = np.log1p(df["value_inr"])
        df["log_weight_kg"] = np.log1p(df["weight_kg"])
        df["log_quantity"] = np.log1p(df["quantity"])

        num_features = [
            "quantity",
            "weight_kg",
            "value_inr",
            "gst_rate",
            "risk_score",
            "log_value_inr",
            "log_weight_kg",
            "log_quantity",
        ]

        cat_features = [
            "origin_city",
            "destination_city",
            "origin_state",
            "destination_state",
            "product_category",
            "supplier_name",
            "carrier_name",
            "truck_type",
            "month",
        ]

        scaler = StandardScaler()
        X_num = scaler.fit_transform(df[num_features].astype("float32"))

        cat_encoders: Dict[str, LabelEncoder] = {}
        X_cat = np.zeros((len(df), len(cat_features)), dtype="int64")
        for idx, col in enumerate(cat_features):
            le = LabelEncoder()
            X_cat[:, idx] = le.fit_transform(df[col].astype(str))
            cat_encoders[col] = le

        logger.info(
            "Preprocessing complete. Num features: %d, Cat features: %d, Delay_reason classes: %d",
            len(num_features),
            len(cat_features),
            len(delay_reason_le.classes_),
        )

        return X_num, X_cat, y_binary, y_reg, y_class, cat_encoders, {
            "scaler": scaler,
            "delay_reason_le": delay_reason_le,
        }, num_features, cat_features

    except Exception as exc:  # noqa: BLE001
        logger.exception("Preprocessing failed.")
        raise DataLoadError("Failed to preprocess training data.") from exc


# -------------------------------------------------------------------
# Training & evaluation
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


def train_multitask_transformer(
    db_path: str = "data/flowsight.duckdb",
    table_exists: bool = True,
    output_dir: str = "models",
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Connecting to DuckDB at %s", db_path)
        conn = duckdb.connect(db_path, read_only=False)
        if not table_exists:
            raise DataLoadError("Expected training_frame table not found (flag set false).")

        df = load_training_frame(conn)
        (
            X_num,
            X_cat,
            y_binary,
            y_reg,
            y_class,
            cat_encoders,
            extra_artifacts,
            num_features,
            cat_features,
        ) = preprocess_data(df)

        X_num_tr, X_num_val, X_cat_tr, X_cat_val, y_bin_tr, y_bin_val, y_reg_tr, y_reg_val, y_cls_tr, y_cls_val = train_test_split(
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

        cat_cardinalities = [len(cat_encoders[col].classes_) for col in cat_features]
        num_delay_reason_classes = len(extra_artifacts["delay_reason_le"].classes_)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        model = MultiTaskTransformer(
            num_num_features=X_num.shape[1],
            cat_cardinalities=cat_cardinalities,
            num_delay_reason_classes=num_delay_reason_classes,
        ).to(device)

        # Losses and optimizer
        criterion_bin = nn.BCEWithLogitsLoss()
        criterion_reg = nn.HuberLoss(delta=1.0)

        # Weight rare delay_reason classes slightly higher (simple heuristic)
        class_counts = np.bincount(y_class)
        inv_freq = 1.0 / np.maximum(class_counts, 1)
        class_weights = (inv_freq / inv_freq.mean()).astype(np.float32)
        criterion_cls = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights).to(device)
        )

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        best_macro_f1 = 0.0
        best_state = None
        max_epochs = 40

        logger.info("Starting training for up to %d epochs...", max_epochs)

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

                # Weight tasks; multiclass slightly down-weighted
                loss = loss_bin + loss_reg + 0.7 * loss_cls
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            metrics = evaluate(model, val_loader, device)
            scheduler.step(metrics["macro_f1"])

            logger.info(
                "Epoch %d | loss=%.4f | val_auc=%.4f | val_mae=%.4f | val_macro_f1=%.4f",
                epoch,
                epoch_loss / len(train_loader),
                metrics["binary_auc"],
                metrics["reg_mae"],
                metrics["macro_f1"],
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_state = model.state_dict()

        if best_state is None:
            raise ModelTrainingError("Training completed but no best model state was stored.")

        model.load_state_dict(best_state)

        model_path = os.path.join(output_dir, "multitask_transformer.pt")
        torch.save(model.state_dict(), model_path)

        # Save preprocessing artifacts
        joblib.dump(extra_artifacts["scaler"], os.path.join(output_dir, "mt_scaler_num.pkl"))
        joblib.dump(cat_encoders, os.path.join(output_dir, "mt_cat_encoders.pkl"))
        joblib.dump(
            extra_artifacts["delay_reason_le"],
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
        # Already logged; re-raise for upstream handling if needed
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during multi-task Transformer training.")
        raise ModelTrainingError("Unexpected failure during multi-task Transformer training.") from exc


if __name__ == "__main__":
    train_multitask_transformer()
