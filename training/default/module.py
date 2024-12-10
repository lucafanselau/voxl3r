from dataclasses import dataclass, field, asdict
from typing import Literal
import torch
import torch.nn as nn
import lightning.pytorch as pl
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)

from experiments.surface_net_3d.visualize import (
    VoxelGridVisualizer,
    VoxelVisualizerConfig,
)
from utils.config import BaseConfig

class BaseLightningModuleConfig(BaseConfig):
    learning_rate: float
    scheduler_factor: float
    scheduler_patience: int
    weight_decay: float

    max_epochs: int
    eta_min: float
    scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"] = "CosineAnnealingLR"


class BaseLightningModule(pl.LightningModule):
    def __init__(self, module_config: BaseLightningModuleConfig, ModelClass: nn.Module, model_config: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["ModelClass"])

        # Store config
        self.config = module_config

        # Initialize the model
        self.model = ModelClass(config=model_config)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
                "auroc": BinaryAUROC(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self, x: Float[Tensor, "batch channels depth height width"]
    ) -> Float[Tensor, "batch 1 depth height width"]:
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y = batch["X"], batch["Y"]
        y_hat = self(x)

        N, C, W, H, D = y.shape
        count_pos = y.sum(dim=(1, 2, 3, 4))
        count_pos[count_pos == 0] = 1
        pos_weight = ((W * H * D - count_pos) / count_pos).reshape(N, 1, 1, 1, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loss = criterion(y_hat, y.float())
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        metrics = self.train_metrics(probs.detach().cpu(), y.detach().int().cpu())

        # Log everything
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)

        return {"loss": loss, "pred": y_hat.detach().cpu()}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        metrics = self.val_metrics(probs.detach().cpu(), y.detach().int().cpu())

        # Log everything
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)

        return {"loss": loss, "pred": y_hat.detach().cpu()}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        metrics = self.test_metrics(probs, y.int())
        
        # Log everything
        self.log("test_loss", loss)
        self.log_dict(metrics)

        return {"loss": loss, "pred": y_hat.detach().cpu()}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler == "ReduceLROnPlateau":
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.config.scheduler_factor,
                    patience=self.config.scheduler_patience,
                    verbose=True,
                ),
                "monitor": "train_loss",
            }
        elif self.config.scheduler == "CosineAnnealingLR":
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.max_epochs,
                    eta_min=self.config.eta_min,
                ),
                "interval": "epoch",
                "frequency": 1,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
