from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
import lightning.pytorch as pl
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


@dataclass
class SurfaceNet3DConfig:
    in_channels: int = 32
    base_channels: int = 32


class SurfaceNet3D(nn.Module):
    def __init__(self, config: SurfaceNet3DConfig):
        super().__init__()

        # Store config
        self.config = config

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(config.in_channels, config.base_channels, 1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(config.base_channels, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(config.base_channels * 2, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(config.base_channels * 2, config.base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 4),
            nn.ReLU(),
            nn.Conv3d(config.base_channels * 4, config.base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 4),
            nn.ReLU(),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(
                config.base_channels * 4, config.base_channels * 2, 2, stride=2
            ),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv3d(config.base_channels * 4, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(config.base_channels * 2, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose3d(
                config.base_channels * 2, config.base_channels, 2, stride=2
            ),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(config.base_channels * 2, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, 1, 1),
        )

    def forward(
        self, x: Float[Tensor, "batch channels depth height width"]
    ) -> Float[Tensor, "batch 1 depth height width"]:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Decoder with skip connections
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        return dec1


@dataclass
class LitSurfaceNet3DConfig:
    model_config: SurfaceNet3DConfig = field(default_factory=SurfaceNet3DConfig)
    learning_rate: float = 1e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    weight_decay: float = 0


class LitSurfaceNet3D(pl.LightningModule):
    def __init__(self, module_config: LitSurfaceNet3DConfig):
        super().__init__()
        self.save_hyperparameters("module_config")

        # Store config
        self.config = module_config

        # Initialize the model
        self.model = SurfaceNet3D(config=module_config.model_config)

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
        x, y, *rest = batch
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
        # metrics = self.train_metrics(probs, y.int())

        # Log everything
        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
        # self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        return loss # {"loss": loss, "pred": y_hat.detach().cpu()}
    
    def on_train_epoch_end(self) -> None:
        # collect garbage
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return super().on_train_epoch_end()


    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        metrics = self.val_metrics(probs, y.int())

        # Log everything
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

        return {"loss": loss, "pred": y_hat.detach().cpu()}

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._shared_step(batch, batch_idx)

        # Calculate metrics
        probs = torch.sigmoid(y_hat)
        metrics = self.test_metrics(probs, y.int())

        def visualize_voxel_grid():
            # visualize a single prediction
            visualizer = VoxelGridVisualizer(VoxelVisualizerConfig())

            grid = torch.stack([1 * (probs[0] > 0.5), y[0].int()], dim=0)

            visualizer.visualize_batch(
                torch.ones_like(grid).repeat(1, 3, 1, 1, 1) * 212,
                grid.squeeze(1),
                labels=["Prediction", "Occupancy"],
            )

        # Log everything
        self.log("test_loss", loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

        return {"loss": loss, "pred": y_hat.detach().cpu()}

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            verbose=True,
            min_lr=6e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
