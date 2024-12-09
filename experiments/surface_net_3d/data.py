from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from einops import rearrange
from loguru import logger
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

from positional_encodings.torch_encodings import PositionalEncoding3D


from datasets.scene import (
    SceneDatasetTransformLoadImages,
)
from datasets.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
from datasets.transforms.projection import project_voxel_grid_to_images_seperate
from utils.chunking import (
    compute_coordinates,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose


config = load_yaml_munch("./utils/config.yaml")



class UnwrapVoxelGridTransform:
    """Transform that unwraps the VoxelGridDataset return tuple to only return feature_grid and occupancy_grid"""

    def __call__(
        self, data: Tuple[torch.Tensor, torch.Tensor, dict], idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_grid, occupancy_grid, _ = data

        return feature_grid, occupancy_grid, idx


@dataclass
class SurfaceNet3DDataConfig(OccChunkDatasetConfig, ColorFeatureGridTransformConfig):
    batch_size: int = 16


class SurfaceNet3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config: SurfaceNet3DDataConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.data_config = data_config
        self.transform = ColorFeatureGridTransform(data_config)

        self.grid_dataset = OccChunkDataset(self.data_config, transform=self.transform)

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def get_in_channels(self):
        return (
            (
                3
                + self.data_config.add_projected_depth
                + self.data_config.add_validity_indicator
                + self.data_config.add_viewing_directing * 3
            )
            * self.data_config.seq_len
            * (1 + self.data_config.concatinate_pe)
        )

        # For now the positional encoding is "added" in the projection
        # + (
        #    self.data_config.pe_channels if self.data_config.pe_enabled else 0
        # )

    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        self.grid_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.grid_dataset,
            [0.6, 0.2, 0.2],
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
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def on_exception(self, exception):
        """
        Handle exceptions during training.
        """
        logger.error(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here
