from dataclasses import dataclass, field, asdict
import math
from typing import List, Literal, Optional, Type, Union
import lightning
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
from networks.attention_net import AttentionNet
from training.bce_weighted import BCEWeighted, BCEWeightedConfig
from training.f1_loss import F1LossWithLogits
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linearly warmup learning rate from 0 (or a small lr_start) to base_lr over warmup_epochs,
    then cosine decay from base_lr down to eta_min over (max_epochs - warmup_epochs) epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Current epoch (starts at 0, then increments each .step())
        current_step = self._step_count

        # Fraction of training completed
        if current_step < self.warmup_steps:
            # -- WARMUP PHASE --
            # Linearly scale from 0 to base_lr over warmup_epochs
            warmup_progress = float(current_step) / float(max(1, self.warmup_steps))
            return [base_lr * warmup_progress for base_lr in self.base_lrs]
        else:
            # -- COSINE ANNEALING PHASE --
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class BaseLightningModuleConfig(BCEWeightedConfig):
    learning_rate: float
    scheduler_factor: float
    scheduler_patience: int
    weight_decay: float

    max_epochs: int
    eta_min: float
    scheduler: Literal[
        "ReduceLROnPlateau", "CosineAnnealingLR", "LinearWarmupCosineAnnealingLR"
    ] = "CosineAnnealingLR"
    use_masked_loss: bool = False
    use_aux_loss: bool = False


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: BaseLightningModuleConfig,
        ModelClass: Union[Type[nn.Module], List[Type[nn.Module]]],
        occGridSampler: Optional[SampleOccGrid] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ModelClass", "occGridSampler"])

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
                # "f1": BinaryF1Score(),
                # "auroc": BinaryAUROC(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.train_metrics_masked = metrics.clone(prefix="masked_train_")
        self.val_metrics_masked = metrics.clone(prefix="masked_val_")
        self.test_metrics_masked = metrics.clone(prefix="masked_test_")

        self.test_precision_recall = BinaryPrecisionRecallCurve()
        self.test_precision_recall_masked = BinaryPrecisionRecallCurve()

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

                if isinstance(module, AttentionNet):
                    self.log("pe_scaler", module.pe_scalar, on_step=True, on_epoch=True)
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
        result = self(batch)
        y_hat = result["Y"]

        y = repeat(y, "B 1 X Y Z -> B (1 N) 1 X Y Z", N=y_hat.shape[1]).to(y_hat)

        mask = batch["occ_mask"].to(y).bool().unsqueeze(1)

        if self.config.use_masked_loss:
            # mask_reshaped = mask.repeat(1, y_hat.shape[1], 1, 1, 1, 1)
            # y[~mask_reshaped] = 0.0
            # y_hat[~mask_reshaped] = 0.0
            # y_hat = y_hat * mask
            loss = self.criterion(y, y_hat, mask=mask)
        else:
            loss = self.criterion(y, y_hat)

        if self.config.use_aux_loss:
            y_hat_surface = result["Y_surface"]
            y_aux = batch["Y"]
            y_aux = repeat(
                y_aux, "B 1 X Y Z -> B (1 N) 1 X Y Z", N=y_hat_surface.shape[1]
            ).to(y_hat_surface)
            loss = 0.7 * loss + 0.3 * self.criterion(y_aux, y_hat_surface, mask=mask)

        if isinstance(self.criterion, BCEWeighted):
            self.log(
                "pos_weight", self.criterion.pos_weight, on_step=True, on_epoch=True
            )

        return loss, y_hat, y, mask

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        loss, y_hat, y, mask = self._shared_step(batch, batch_idx)

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
        loss, y_hat, y, mask = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        self.val_metrics(probs, y.int())

        # Log everything
        self.log("val_loss", loss.mean().item(), on_step=True, on_epoch=True)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True)

        return {
            "loss": loss.mean(),
            "loss_batch": loss,
            "pred": y_hat.detach().cpu(),
            "y": y.detach().cpu(),
        }

    def test_step(self, batch, batch_idx):
        loss, y_hat, y, mask = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        self.test_metrics(probs, y.int())
        self.test_metrics_masked(probs[mask], y[mask].int())

        # Log everything
        self.log("test_loss", loss.mean().item(), on_step=True, on_epoch=True)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.test_metrics_masked, on_step=True, on_epoch=True)

        self.test_precision_recall.update(probs, y.int())
        self.test_precision_recall_masked.update(probs[mask], y[mask].int())

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

        fig_masked, _ = self.test_precision_recall_masked.plot(score=True)
        fig_masked.savefig("pr_curve_masked.png")

        self.test_precision_recall.reset()
        self.test_precision_recall_masked.reset()

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
        elif self.config.scheduler == "LinearWarmupCosineAnnealingLR":
            # based on nanoGPT
            lr_scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_steps=10946 // (self.config.batch_size * self.config.accumulate_grad_batches),
                    max_steps=30 * 1000,
                    eta_min=self.config.eta_min,
                ),
                "interval": "step",
                "frequency": 1,
            }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
