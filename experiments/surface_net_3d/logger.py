import io
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple
from einops import rearrange
from lightning import Callback, LightningModule, Trainer
from torch import Tensor
from lightning.pytorch.loggers import WandbLogger
import torch
from jaxtyping import Float
import trimesh
import wandb

from experiments.mast3r_3d.data import Mast3r3DDataModule
from experiments.surface_net_3d.data import SurfaceNet3DDataModule
from utils.data_parsing import load_yaml_munch

config = load_yaml_munch("./utils/config.yaml")
# Pytorch Lightning Callback to log the 3D voxel grids at end of epoch
class VoxelGridLoggerCallback(Callback):
    # we also store the batch idx to know which batch is which
    # Tuple is [pred, batch_idx, gt, dataset_idx]
    results: Mapping[str, Optional[Tuple[Tensor, int, Tensor, int]]] = {
        "train": None,
        "val": None,
        "test": None,
    }

    def __init__(self, wandb: WandbLogger, n_epochs: Tuple[int, int, int] = (5, 5, 1)):
        self.wandb = wandb
        self.n_epochs = {"train": n_epochs[0], "val": n_epochs[1], "test": n_epochs[2]}

    # use the on_batch_end to store the first result of the epoch
    def sink_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        type: str,
    ) -> None:
        # outputs is a dict with the loss and the predictions
        if self.results[type] is None:
            self.results[type] = (
                outputs["pred"][0].detach(),
                batch_idx,
                batch[1][0].detach(),
                batch[2][0].item(),
            )

    def on_train_batch_end(self, *args, **kwargs) -> None:
        kwargs["type"] = "train"
        self.sink_on_batch_end(*args, **kwargs)

    def on_validation_batch_end(self, *args, **kwargs) -> None:
        kwargs["type"] = "val"
        self.sink_on_batch_end(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs) -> None:
        kwargs["type"] = "test"
        self.sink_on_batch_end(*args, **kwargs)

    def create_voxel_grid(
        self,
        grid: Float[Tensor, "1 X Y Z"],
        occ_threshold: float = 0.5,
        idx: int = 0,
    ) -> wandb.Object3D:
        # grid is a tensor of shape (1, X, Y, Z)

        # create a trimesh object for this grid
        occ = grid
        _1, X, Y, Z = occ.shape

        encoding = trimesh.voxel.encoding.DenseEncoding(
            (occ > occ_threshold).detach().squeeze(0).bool().cpu().numpy()
        )

        grid = trimesh.voxel.VoxelGrid(encoding)
        occ_re = rearrange(occ, "1 X Y Z -> X Y Z 1")
        # visualize the occ as a opacity
        colors = occ_re.repeat(1, 1, 1, 4).float()
        if (occ > 0.5).sum() == 0:
            return None, None
        mesh = grid.as_boxes(colors=colors.detach().cpu().numpy())

        tmp_dir = Path(self.wandb.experiment.dir) / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        mesh.export(str(tmp_dir / f"voxel_grid_{idx}.glb"))
        object_3d = wandb.Object3D(str(tmp_dir / f"voxel_grid_{idx}.glb"))
        # delete the tmp file
        file_name = tmp_dir / f"voxel_grid_{idx}.glb"
        return object_3d, file_name

    def log_results(self, trainer: Trainer, type: str):
        pred, _, gt, idx = self.results[type]

        object_3d_pred, file_name_pred = self.create_voxel_grid(pred, idx=0)
        object_3d_gt, file_name_gt = self.create_voxel_grid(gt, idx=1)

        if object_3d_pred is None:
            return

        # also let's log the first image
        # trainer.datamodule.
        datamodule: SurfaceNet3DDataModule | Mast3r3DDataModule = trainer.datamodule
        if isinstance(datamodule, Mast3r3DDataModule):
            # because this one doesn't have a transform
            dict = datamodule.mast3r_grid_dataset.get_at_idx(idx)
        else:
            _features, _gt, dict = datamodule.grid_dataset.get_at_idx(idx)
        image_path = dict["images"][0][0]
        image_path = str(Path(config.data_dir) / Path(*Path(image_path).parts[Path(image_path).parts.index("data") + 3 :]))

        self.wandb.experiment.log(
            {
                f"{type}_prediction": object_3d_pred,
                f"{type}_ground_truth": object_3d_gt,
                f"{type}_image": wandb.Image(image_path),
            }
        )
        file_name_pred.unlink()
        file_name_gt.unlink()

    # For now this is down only at the end of the training epoch
    def on_train_epoch_end(self, trainer, pl_module):
        if self.results["train"] is None:
            return
        if (trainer.current_epoch + 1) % self.n_epochs["train"] == 0:
            self.log_results(trainer, "train")
        # reset the train results
        self.results["train"] = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.results["val"] is None:
            return
        if (trainer.current_epoch + 1) % self.n_epochs["val"] == 0:
            self.log_results(trainer, "val")
        # reset the val results
        self.results["val"] = None

    def on_test_epoch_end(self, trainer, pl_module):
        if self.results["test"] is None:
            return
        if (trainer.current_epoch + 1) % self.n_epochs["test"] == 0:
            self.log_results(trainer, "test")
        # reset the test results
        self.results["test"] = None
