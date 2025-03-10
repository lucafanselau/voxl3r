from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Type, Union
from lightning.pytorch.utilities import grad_norm
from einops import repeat
import torch
import torch.nn as nn
import lightning.pytorch as pl
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchmetrics import MetricCollection, PrecisionRecallCurve
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryPrecisionRecallCurve,
)

from datasets.transforms_batched.sample_occ_grid import SampleOccGrid
from training.bce_weighted import BCEWeighted, BCEWeightedConfig
from training.f1_loss import F1LossWithLogits


class BaseLightningModuleConfig(BCEWeightedConfig):
    learning_rate: float
    scheduler_factor: float
    scheduler_patience: int
    weight_decay: float

    max_epochs: int
    eta_min: float
    scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"] = "CosineAnnealingLR"


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: BaseLightningModuleConfig,
        ModelClass: Union[Type[nn.Module], List[Type[nn.Module]]],
        occGridSampler: Optional[SampleOccGrid] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ModelClass"])

        # Store config
        self.config = config

        # Initialize the model
        if isinstance(ModelClass, list):
            self.model = nn.ModuleList(
                [model_class(config=config) for model_class in ModelClass]
            )
        else:
            self.model = ModelClass(config=config)

        # Loss function
        self.criterion = BCEWeighted(self.config)  # F1LossWithLogits()  #

        # Metrics
        metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
                # "auroc": BinaryAUROC(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.test_precision_recall = BinaryPrecisionRecallCurve()

        self.occGridSampler = occGridSampler

    def forward(
        self,
        batch: dict,
    ) -> Float[Tensor, "batch 1 depth height width"]:

        if isinstance(self.model, nn.ModuleList):
            import time

            start_time = time.time()
            for i, module in enumerate(self.model):
                batch = module(batch)
                end_time = time.time()
                self.log(
                    f"time_taken_{i}",
                    end_time - start_time,
                    on_step=True,
                    on_epoch=True,
                )
                start_time = end_time
            return batch
        else:
            y_hat = self.model(batch)
        return y_hat

    def _shared_step(self, batch, batch_idx):
        if self.occGridSampler is not None:
            batch = self.occGridSampler(batch)

        y: torch.Tensor = batch["Y"]
        y_hat = self(batch)["Y"]

        y = repeat(y, "B 1 X Y Z -> B (1 N) 1 X Y Z", N=y_hat.shape[1]).to(y_hat)
        loss = self.criterion(y, y_hat)

        if isinstance(self.criterion, BCEWeighted):
            self.log(
                "pos_weight", self.criterion.pos_weight, on_step=True, on_epoch=True
            )

        return loss, y_hat, y

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        self.train_metrics(probs, y.int())

        # Log everything
        self.log(
            "train_loss", loss.mean().item(), prog_bar=True, on_step=True, on_epoch=True
        )
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)

        return {"loss": loss.mean(), "loss_batch": loss, "pred": y_hat.detach().cpu()}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        self.val_metrics(probs, y.int())

        # Log everything
        self.log("val_loss", loss.mean().item(), on_step=True, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)

        return {"loss": loss.mean(), "loss_batch": loss, "pred": y_hat.detach().cpu()}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        self.test_metrics(probs, y.int())

        # Log everything
        self.log("test_loss", loss.mean().item(), on_step=True, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)

        self.test_precision_recall.update(probs, y.int())

        # y_true = y.int().flatten()
        # y_pred = probs.flatten().unsqueeze(1)

        # # y_pred needs to be a mapping (N, 2) where the first is the probability of the zero class and the second is the probability of the one class
        # y_pred = torch.cat((1 - y_pred, y_pred), dim=1)

        # curve = wandb.plot.pr_curve(y_true=y_true.cpu(), y_probas=y_pred.cpu(), labels=["0", "1"])
        # self.logger.experiment.log({"pr_curve": curve})

        return {"loss": loss.mean(), "loss_batch": loss, "pred": y_hat.detach().cpu()}

    def on_before_optimizer_step(self, optimizer):
        norm_order = 2.0
        norms = grad_norm(self, norm_type=norm_order)
        self.log(
            "Total gradient (norm)",
            norms[f"grad_{norm_order}_norm_total"],
            on_step=True,
            on_epoch=False,
        )

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        fig, ax = self.test_precision_recall.plot(score=True)

        # store as image
        fig.savefig("pr_curve.png")

        self.test_precision_recall.reset()

    def configure_optimizers(self):
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
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
