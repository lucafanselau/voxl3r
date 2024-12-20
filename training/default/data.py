from typing import Optional

import torch
from utils.config import BaseConfig
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split

class DefaultDataModuleConfig(BaseConfig):
    batch_size: int = 16
    num_workers: int = 11


class DefaultDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DefaultDataModuleConfig, dataset: Dataset):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        self.data_config = data_config
        self.dataset = dataset
        # Will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        self.dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [0.7, 0.2, 0.1],
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
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            # pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            # pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            # pin_memory=True,
        )

    def on_exception(self, exception):
        """
        Handle exceptions during training.
        """
        print(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here