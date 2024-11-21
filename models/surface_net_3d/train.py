import glob
import os
from pathlib import Path
import time
from einops import rearrange
import torch
import lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from jaxtyping import Float

from dataset import SceneDataset, SceneDatasetTransformToTorch
from models.surface_net_3d.model import (
    LitSurfaceNet3D,
    LitSurfaceNet3DConfig,
    SurfaceNet3DConfig,
)
from models.surface_net_3d.data import SurfaceNet3DDataConfig, SurfaceNet3DDataModule
from models.surface_net_3d.logger import VoxelGridLoggerCallback
from models.surface_net_3d.visualize import (
    VoxelVisualizerConfig,
    visualize_voxel_grids,
)
from models.surface_net_baseline.data import project_points_to_images
from utils.chunking import create_chunk, mesh_2_voxels
from utils.data_parsing import load_yaml_munch
from utils.visualize import visualize_mesh

config = load_yaml_munch("./utils/config.yaml")


def visualize_unprojection(data):
    transform = SceneDatasetTransformToTorch("cuda")
    image_names, camera_params_list = data["images"]
    images, transformations, points, gt = transform.forward(data)
    # and normalize images
    images = images / 255.0

    # reshape points beforehand
    X = project_points_to_images(points, images, transformations)

    rgb_list = rearrange(
        X, "i c w h d -> (w h d) i c"
    )  # rearrange(X, "p (i c) -> p i c", c=3)
    points = rearrange(points, "c w h d -> (w h d) c")
    gt = gt.reshape(-1)
    mask = rgb_list != -1
    denom = torch.sum(torch.sum(mask, -1) / 3, -1)
    rgb_list[rgb_list == -1.0] = 0.0
    rgb_list_pruned = rgb_list[denom != 0]
    points_pruned = points[denom != 0]
    occ = gt[denom != 0]
    denom = denom[denom != 0]
    rgb_list_avg = torch.sum(rgb_list_pruned, dim=1) / denom.unsqueeze(-1).repeat(1, 3)

    visualize_mesh(
        data["mesh"],
        point_coords=points_pruned.cpu().numpy(),
        images=image_names,
        camera_params_list=camera_params_list,
        heat_values=occ.cpu().numpy(),
        rgb_list=rgb_list_avg.cpu().numpy(),
    )


if __name__ == "__main__":

    torch.set_float32_matmul_precision("medium")

    data_config = SurfaceNet3DDataConfig(data_dir=config.data_dir, batch_size=16, num_workers=11, scenes=load_yaml_munch(Path("./data") / "dslr_undistort_config.yml").scene_ids)
    datamodule = SurfaceNet3DDataModule(data_config=data_config)

    # Create configs
    model_config = SurfaceNet3DConfig(
        in_channels=datamodule.get_in_channels(), base_channels=32
    )
    lit_config = LitSurfaceNet3DConfig(
        model_config=model_config,
        learning_rate=1e-4,
        scheduler_factor=0.5,
        scheduler_patience=5,
    )

    # Initialize model and datamodule
    model = LitSurfaceNet3D(module_config=lit_config)
    # model = torch.compile(model)

    # Setup logging
    logger = WandbLogger(
        project="surface-net-3d", save_dir="./.lightning/surface-net-3d"
    )

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Checkpointing callbacks
    filename = "{epoch}-{step}"
    callbacks = [
        ModelCheckpoint(
            filename=filename + f"-{{val_{name}:.2f}}",
            monitor=f"val_{name}",
            save_top_k=3,
            mode=mode,
        )
        for [name, mode] in [
            ["loss", "min"],
            ["accuracy", "max"],
            ["f1", "max"],
            ["auroc", "max"],
        ]
    ]

    # Save the model every 5 epochs
    every_five_epochs = ModelCheckpoint(
        every_n_epochs=5,
        save_top_k=-1,
        save_last=True,
    )

    # Custom callback for logging the 3D voxel grids
    voxel_grid_logger = VoxelGridLoggerCallback(wandb=logger)

    # Initialize trainer
    trainer = Trainer(
        max_epochs=1000,
        log_every_n_steps=4,
        callbacks=[*callbacks, every_five_epochs, lr_monitor, voxel_grid_logger],
        logger=logger,
        precision="bf16-mixed",
        default_root_dir="./.lightning/surface-net-3d",
    )

    # Train
    # get last created folder in ./.lightning/surface-net-3d/surface-net-3d/
    # while True:
    #     try:

    #         # "61psgsi5"

    #         ckpt_folder = list(
    #             Path("./.lightning/surface-net-3d/surface-net-3d/").glob("*")
    #         )
    #         ckpt_folder = sorted(ckpt_folder, key=os.path.getmtime)
    #         last_ckpt_folder = Path(
    #             "./.lightning/surface-net-3d/surface-net-3d/61psgsi5"
    #         )  # ckpt_folder[-1]
    #         print(f"Resuming training from {last_ckpt_folder}")
    #         trainer.fit(
    #             model,
    #             datamodule=datamodule,
    #             ckpt_path=last_ckpt_folder / "checkpoints/last.ckpt",
    #         )

    #     except KeyboardInterrupt:
    #         break
    #     except Exception as e:
    #         print(f"ALAAAAAARM {e}")
    #         # sleep for 10 seconds
    #         time.sleep(10)

    trainer.fit(model, datamodule=datamodule)

    # Save best checkpoints info
    base_path = Path(callbacks[0].best_model_path).parents[1]
    result = base_path / "best_ckpts.pt"
    result_dict = {
        "best_model_val_loss": callbacks[0].best_model_path,
        "best_model_val_accuracy": callbacks[1].best_model_path,
        "best_model_val_f1": callbacks[2].best_model_path,
        "best_model_val_auroc": callbacks[3].best_model_path,
        "last_model_path": every_five_epochs.best_model_path,
    }
    torch.save(result_dict, result)
