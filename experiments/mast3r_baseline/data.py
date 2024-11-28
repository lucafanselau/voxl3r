# Transform to create

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from experiments.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
import lightning.pytorch as pl

from torch.utils.data import DataLoader, random_split


@dataclass
class Mast3rBaselineDataTransformConfig:
    pass


class Mast3rBaselineDataTransform:
    """
    Transform occ chunk data to be used for inputting into master (eg. image pairs)

    Additionally we also output the transforms, for the fixed alignment strategy
    """

    def __init__(self, config: Mast3rBaselineDataTransformConfig):
        self.config = config

    def __call__(
        self, data: Tuple[torch.Tensor, dict], idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


@dataclass
class Mast3rBaselineDataConfig(OccChunkDatasetConfig):
    batch_size: int = 16


class Mast3rBaselineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_config: Mast3rBaselineDataConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform"])
        self.data_config = data_config
        self.transform = Mast3rBaselineDataTransform(data_config)

        self.grid_dataset = OccChunkDataset(self.data_config, transform=self.transform)

        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
        print(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here
