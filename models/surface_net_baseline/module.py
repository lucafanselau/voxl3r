import inspect
from typing import Tuple
import lightning as pl
from dataclasses import asdict, dataclass

import torch
import torchmetrics

from models.surface_net_baseline.model import SimpleOccNet, SimpleOccNetConfig


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)


@dataclass
class LRConfig:
    T_max: int = 10


class OccSurfaceNet(pl.LightningModule):
    def __init__(
        self,
        config: SimpleOccNetConfig,
        optimizer_config: OptimizerConfig,
        lr_config: LRConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.model = SimpleOccNet(config)

        # metrics
        self.train_accuracy = torchmetrics.Accuracy(task="binary")
        self.val_accuracy = torchmetrics.Accuracy(task="binary")

        # test metrics
        self.test_accuracy = torchmetrics.Accuracy(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        pred = self.model(X)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, Y)
        self.log("train_loss", loss)

        # compute metrics with torchmetrics
        acc = self.train_accuracy(pred, Y)
        self.log("train_accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        X, Y = batch
        pred = self.model(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, Y)
        self.log("val_loss", loss)

        # compute metrics with torchmetrics
        acc = self.val_accuracy(pred, Y)
        self.log("val_accuracy", acc)
        return loss
    
    def test_step(self, batch, batch_idx) -> torch.Tensor | None:
        X, Y = batch
        pred = self.model(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, Y)
        self.log("test_loss", loss)
        # compute metrics with torchmetrics
        metrics = self.compute_metrics(pred, Y)
        self.log_dict(metrics)
        return loss
    
    def compute_metrics(self, pred, Y) -> dict[str, float]:
        metrics = {}

        # Apply sigmoid to get probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Calculate metrics
        metrics["accuracy"] = self.test_accuracy(pred_probs, Y)
        metrics["precision"] = self.test_precision(pred_probs, Y)
        metrics["recall"] = self.test_recall(pred_probs, Y)
        metrics["f1"] = self.test_f1(pred_probs, Y)
        metrics["auroc"] = self.test_auroc(pred_probs, Y)
        
        # Add test/ prefix to all metrics
        metrics = {f"test/{k}": v for k, v in metrics.items()}
        
        return metrics

    def configure_optimizers(self):
        optimizer_cfg = self.hparams.optimizer_config
        learning_rate = optimizer_cfg.learning_rate
        weight_decay = optimizer_cfg.weight_decay
        betas = optimizer_cfg.betas
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device.type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **asdict(self.hparams.lr_config)
            ),
            "interval": "step",
            "name": "CosineAnnealingLR - Scheduler",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
