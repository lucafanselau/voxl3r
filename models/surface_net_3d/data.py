from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split, Dataset
from jaxtyping import Float, Int, Bool
from torch import Tensor
import numpy as np
import trimesh

from dataset import SceneDataset


@dataclass
class SurfaceNet3DDataConfig:
    data_dir: str = "datasets/scannetpp/data"
    batch_size: int = 32
    num_workers: int = 4
    camera: str = "iphone"
    train_val_split: float = 0.9
    scenes: Optional[List[str]] = None
    grid_resolution: float = 0.02
    grid_size: Tuple[int, int, int] = (64, 64, 64)
    fixed_idx: Optional[int] = None


def convert_to_voxel_grid(
    mesh: trimesh.Trimesh,
    image_paths: Path,
    camera_params: List[dict],
    grid_size: Tuple[int, int, int],
    resolution: float = 0.02,
) -> Tuple[
    Float[Tensor, "channels height width depth"], Bool[Tensor, "height width depth"]
]:
    """
    Convert mesh and image data to voxel grid representation.

    Args:
        mesh: Trimesh mesh object
        image_paths: Path to image directory
        camera_params: List of camera parameter dictionaries
        grid_size: Size of the voxel grid (H, W, D)
        resolution: Voxel grid resolution

    Returns:
        Tuple containing:
            - features: (C, H, W, D) tensor of voxel features
            - occupancy: (H, W, D) tensor of boolean occupancy values
    """
    # TODO in this file
    # 1. Construct voxel grids for a set number of images
    # .  - These will have the be aligned to the chunk corresponding to

    H, W, D = grid_size

    # Get mesh bounds
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]

    # Create regular grid
    x = torch.linspace(bounds_min[0], bounds_max[0], H)
    y = torch.linspace(bounds_min[1], bounds_max[1], W)
    z = torch.linspace(bounds_min[2], bounds_max[2], D)

    # Create grid points
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    # TODO: Implement actual feature extraction from images
    # For now, return dummy features (32 channels as specified in model)
    features = torch.randn(32, H, W, D)

    # TODO: Implement actual occupancy computation from mesh
    # For now, return random occupancy
    occupancy = torch.randint(0, 2, (H, W, D), dtype=torch.bool)

    return features, occupancy


class VoxelGridDataset(Dataset):
    def __init__(self, base_dataset: SceneDataset, config: SurfaceNet3DDataConfig):
        self.base_dataset = base_dataset
        self.config = config

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(
        self, idx
    ) -> Tuple[
        Float[Tensor, "channels height width depth"], Bool[Tensor, "height width depth"]
    ]:
        data = self.base_dataset[
            self.config.fixed_idx if self.config.fixed_idx is not None else idx
        ]

        features, occupancy = convert_to_voxel_grid(
            mesh=data["mesh"],
            image_paths=data["path_images"],
            camera_params=data["camera_params"],
            grid_size=self.config.grid_size,
            resolution=self.config.grid_resolution,
        )

        return features, occupancy


class SurfaceNet3DDataModule(pl.LightningModule):
    def __init__(self, config: SurfaceNet3DDataConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        # SceneDataset handles data preparation internally
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """
        # Create base dataset
        base_dataset = SceneDataset(
            camera=self.config.camera,
            data_dir=self.config.data_dir,
            scenes=self.config.scenes,
        )

        # Wrap with VoxelGridDataset
        full_dataset = VoxelGridDataset(base_dataset, self.config)

        # Calculate split sizes (60%, 20%, 20%)
        n_samples = len(full_dataset)
        n_train = int(n_samples * 0.6)
        n_val = int(n_samples * 0.2)
        n_test = n_samples - n_train - n_val

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        # Assign datasets based on stage
        if stage == "fit" or stage is None:
            pass  # datasets already assigned

        if stage == "test" or stage is None:
            pass  # test dataset already assigned

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )
