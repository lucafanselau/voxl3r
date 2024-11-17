from pathlib import Path
from einops import rearrange
import torch
import lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset import SceneDataset, SceneDatasetTransformToTorch
from models.surface_net_3d.model import (
    LitSurfaceNet3D,
    LitSurfaceNet3DConfig,
    SurfaceNet3DConfig,
)
from models.surface_net_3d.data import (
    SurfaceNet3DDataConfig,
    SurfaceNet3DDataModule
)
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

    rgb_list = rearrange(X, "i c w h d -> (w h d) i c")#rearrange(X, "p (i c) -> p i c", c=3)
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

    # Create configs
    model_config = SurfaceNet3DConfig(in_channels=32, base_channels=32)

    lit_config = LitSurfaceNet3DConfig(
        model_config=model_config,
        learning_rate=1e-3,
        scheduler_factor=0.5,
        scheduler_patience=5,
    )

    data_config = SurfaceNet3DDataConfig(
        data_dir=config.data_dir,
        batch_size=2
    )

    # Initialize model and datamodule
    model = LitSurfaceNet3D(config=lit_config)
    datamodule = SurfaceNet3DDataModule(config=data_config)

    # Prepare data and setup
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get a single batch from the training dataset
    train_loader = datamodule.train_dataloader()
    features, occupancy = next(iter(train_loader))

    # Visualize features (first sample from batch)
    vis_config = VoxelVisualizerConfig(
        opacity=0.6,
        show_edges=True,
        cmap="viridis",
        window_size=(1200, 800),
        camera_position=(100, 100, 100),
    )

    # Visualize feature channels
    print("Visualizing feature channels...")
    visualize_voxel_grids(
        features[0],  # First sample from batch
        config=vis_config,
        save_path="feature_channels.png",
    )

    # Visualize occupancy
    print("Visualizing occupancy...")
    vis_config.cmap = "binary"  # Better for binary data
    visualize_voxel_grids(
        occupancy[0].unsqueeze(0),  # Add channel dimension
        config=vis_config,
        save_path="occupancy.png",
    )

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

    # Initialize trainer
    trainer = Trainer(
        max_epochs=100,
        log_every_n_steps=4,
        callbacks=[*callbacks, every_five_epochs, lr_monitor],
        logger=logger,
        precision="bf16-mixed",
        default_root_dir="./.lightning/surface-net-3d",
    )

    # Train
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
