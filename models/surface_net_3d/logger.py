import io
from typing import Any, Mapping, Optional, Tuple
from einops import rearrange
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from lightning.pytorch.loggers import WandbLogger
import torch
import trimesh
import wandb


# Pytorch Lightning Callback to log the 3D voxel grids at end of epoch
class VoxelGridLoggerCallback(Callback):
    # we also store the batch idx to know which batch is which
    train_results: Optional[Tuple[Tensor, int]] = None
    val_results: Optional[Tuple[Tensor, int]] = None
    test_results: Optional[Tuple[Tensor, int]] = None

    def __init__(self, wandb: WandbLogger):
        self.wandb = wandb
        self.train_results = None
        self.val_results = None
        self.test_results = None

    # use the on_batch_end to store the first result of the epoch
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # outputs is a dict with the loss and the predictions
        if self.train_results is None:
            self.train_results = (outputs["pred"], batch_idx)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.val_results is None:
            self.val_results = (outputs["pred"], batch_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.test_results is None:
            self.test_results = (outputs["pred"], batch_idx)

    def create_voxel_grid(self, pred: Tensor, batch_idx: int) -> wandb.Object3D:
        # pred is a tensor of shape (batch_size, 1, X, Y, Z)

        # create a trimesh object for this pred
        used = pred[0]
        occ = torch.sigmoid(used)
        _1, X, Y, Z = occ.shape

        encoding = trimesh.voxel.encoding.DenseEncoding(
            torch.ones(X, Y, Z, requires_grad=False).bool().cpu().numpy()
        )

        grid = trimesh.voxel.VoxelGrid(encoding)
        occ_re = rearrange(occ, "1 X Y Z -> X Y Z 1")
        # visualize the occ as a opacity
        colors = torch.zeros((X, Y, Z, 4), requires_grad=False)
        colors[:, :, :, 3] = occ_re.squeeze(-1)
        mesh = grid.as_boxes(colors=colors.detach().cpu().numpy())
        mesh_str = mesh.export(file_type="obj")

        # create fake file for mesh_str
        with io.StringIO() as f:
            f.write(mesh_str)
            f.seek(0)
            return wandb.Object3D(f, file_type="obj")

    def log_voxel_grid(self, results: Tuple[Tensor, int], name: str):
        object_3d = self.create_voxel_grid(results[0], results[1])
        self.wandb.experiment.log(
            {
                f"{name}_voxel_grid": object_3d,
            }
        )

    # For now this is down only at the end of the training epoch
    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_results is None:
            return
        self.log_voxel_grid(self.train_results, "train")
        # reset the train results
        self.train_results = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.val_results is None:
            return
        self.log_voxel_grid(self.val_results, "val")
        # reset the val results
        self.val_results = None

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_results is None:
            return
        self.log_voxel_grid(self.test_results, "test")
        # reset the test results
        self.test_results = None
