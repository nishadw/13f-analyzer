"""
Step 4: Temporal Fusion Transformer (TFT) predictive engine.

Uses pytorch-forecasting's TemporalFusionTransformer which natively handles:
  - Static covariates (fund CIK / strategy metadata)
  - Known past inputs (previous quarter 13F weights)
    - Dynamic continuous inputs (weekly market proxy flows, FRED macro, conviction scores)

The model predicts the portfolio weight distribution for each asset over the
next quarter, and a simplex projection layer enforces the portfolio constraints.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor

import config
from model.simplex_projector import (
    SimplexProjectionLayer,
    tracking_error_loss,
    portfolio_entropy_regularizer,
)


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_time_series_dataset(
    features: pd.DataFrame,
    target_cols: list[str],
    static_cols: list[str],
    time_varying_known_cols: list[str],
    time_varying_unknown_cols: list[str],
    group_col: str = "dummy_group",
    time_col: str = "time_idx",
) -> TimeSeriesDataSet:
    """
    Construct a pytorch-forecasting TimeSeriesDataSet from the feature store.
    Caller must ensure target_cols is non-empty and group_col exists in features.
    """
    if not target_cols:
        raise ValueError("target_cols is empty — trainer must provide at least one target column.")

    df = features.copy().reset_index()
    if "week_ending" in df.columns:
        df = df.rename(columns={"week_ending": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df[time_col] = (df["date"] - df["date"].min()).dt.days // 7

    # Ensure the group column is present
    if group_col not in df.columns:
        df[group_col] = "fund_0"

    # Ensure static categoricals are cast to string so TFT treats them as categories
    for col in static_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop unknown cols that are also already the target or group to avoid duplicates
    safe_unknown = [
        c for c in time_varying_unknown_cols
        if c not in target_cols and c != group_col and c != time_col and c in df.columns
    ]

    primary_target = target_cols[0]
    n_rows = len(df)

    # Scale window sizes so encoder + prediction always fits inside n_rows
    # with a comfortable margin of at least 5 rows for valid training samples.
    max_pred = min(config.TFT_MAX_PREDICTION_LENGTH, max(1, n_rows // 8))
    max_enc  = min(config.TFT_MAX_ENCODER_LENGTH,  max(2, n_rows - max_pred - 5))
    min_enc  = max(1, max_enc // 4)

    # Fill any remaining NaN / inf values — TFT requires fully finite inputs
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].ffill().bfill().fillna(0.0)
    df.replace([float("inf"), float("-inf")], 0.0, inplace=True)

    dataset = TimeSeriesDataSet(
        df,
        time_idx=time_col,
        target=primary_target,
        group_ids=[group_col],
        min_encoder_length=min_enc,
        max_encoder_length=max_enc,
        max_prediction_length=max_pred,
        static_categoricals=[c for c in static_cols if c in df.columns],
        time_varying_known_reals=[c for c in time_varying_known_cols if c in df.columns],
        time_varying_unknown_reals=safe_unknown,
        target_normalizer=GroupNormalizer(groups=[group_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset, df


# ── Lightning Module with portfolio loss ─────────────────────────────────────

class PortfolioTFT(pl.LightningModule):
    """
    Wraps the pytorch-forecasting TFT with a tracking-error loss and applies
    simplex projection to the output so predictions are valid portfolio weights.
    """

    def __init__(
        self,
        n_assets: int,
        time_series_dataset: TimeSeriesDataSet,
        learning_rate: float = config.TFT_LEARNING_RATE,
        entropy_lambda: float = 0.005,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["time_series_dataset"])
        self.n_assets = n_assets
        self.entropy_lambda = entropy_lambda
        self.projection = SimplexProjectionLayer()

        self.tft = TemporalFusionTransformer.from_dataset(
            time_series_dataset,
            learning_rate=learning_rate,
            hidden_size=config.TFT_HIDDEN_SIZE,
            attention_head_size=config.TFT_ATTENTION_HEAD_SIZE,
            dropout=config.TFT_DROPOUT,
            hidden_continuous_size=config.TFT_HIDDEN_CONTINUOUS_SIZE,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

    def forward(self, x: dict) -> Tensor:
        output = self.tft(x)
        # output.prediction: (batch, horizon, n_quantiles)
        # Use the median (quantile 0.5 index) as point prediction
        pred = output.prediction[..., output.prediction.shape[-1] // 2]  # (batch, horizon)
        # For multi-asset: flatten horizon into asset dim and project
        if pred.shape[-1] >= self.n_assets:
            weights = pred[..., : self.n_assets]
        else:
            weights = pred
        return self.projection(weights)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        pred = self(x)
        target = y[0].squeeze(-1)
        # Align shapes
        min_len = min(pred.shape[-1], target.shape[-1])
        loss = tracking_error_loss(pred[..., :min_len], target[..., :min_len])
        loss = loss + portfolio_entropy_regularizer(pred, self.entropy_lambda)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        pred = self(x)
        target = y[0].squeeze(-1)
        min_len = min(pred.shape[-1], target.shape[-1])
        loss = tracking_error_loss(pred[..., :min_len], target[..., :min_len])
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def predict_portfolio(self, x: dict) -> Tensor:
        """Run inference and return simplex-projected portfolio weights."""
        self.eval()
        with torch.no_grad():
            return self(x)


# ── Training entry point ──────────────────────────────────────────────────────

def train_model(
    features: pd.DataFrame,
    target_cols: list[str],
    static_cols: list[str],
    time_varying_known_cols: list[str],
    time_varying_unknown_cols: list[str],
    group_col: str = "dummy_group",
    checkpoint_dir: Path | None = None,
) -> PortfolioTFT:
    checkpoint_dir = checkpoint_dir or config.MODEL_DIR

    dataset, df = prepare_time_series_dataset(
        features=features,
        target_cols=target_cols,
        static_cols=static_cols,
        time_varying_known_cols=time_varying_known_cols,
        time_varying_unknown_cols=time_varying_unknown_cols,
        group_col=group_col,
    )

    train_ds = dataset.from_dataset(dataset, df, stop_randomization=True)
    val_ds = dataset.from_dataset(dataset, df, predict=True)

    train_loader = train_ds.to_dataloader(
        train=True, batch_size=config.TFT_BATCH_SIZE, num_workers=0
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=config.TFT_BATCH_SIZE, num_workers=0
    )

    model = PortfolioTFT(
        n_assets=len(target_cols),
        time_series_dataset=dataset,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, mode="min"),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=config.TFT_MAX_EPOCHS,
        callbacks=callbacks,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        log_every_n_steps=5,
        accelerator="cpu",  # pytorch-forecasting attention masks are CPU-only; MPS causes device mismatch
    )

    trainer.fit(model, train_loader, val_loader)

    dataset_path = checkpoint_dir / "dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

    return model


_DATASET_CACHE: TimeSeriesDataSet | None = None


def load_best_checkpoint(checkpoint_dir: Path | None = None) -> Optional[PortfolioTFT]:
    global _DATASET_CACHE
    checkpoint_dir = checkpoint_dir or config.MODEL_DIR
    ckpts = sorted(checkpoint_dir.glob("tft-*.ckpt"))
    if not ckpts:
        return None
    def _val_loss(p: Path) -> float:
        raw = p.stem.split("val_loss=")[-1]
        return float(raw.split("-")[0])

    best = sorted(ckpts, key=_val_loss)[0]

    dataset_path = checkpoint_dir / "dataset.pkl"
    if _DATASET_CACHE is None and dataset_path.exists():
        with open(dataset_path, "rb") as f:
            _DATASET_CACHE = pickle.load(f)

    if _DATASET_CACHE is None:
        raise FileNotFoundError(
            f"dataset.pkl not found at {dataset_path}. Re-run training to regenerate it."
        )

    print(f"[model] loading checkpoint: {best}")
    return PortfolioTFT.load_from_checkpoint(str(best), time_series_dataset=_DATASET_CACHE)
