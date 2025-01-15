from functools import partial
import os
from typing import Optional

import torch
from utils.config import BaseConfig
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
from loguru import logger

class DefaultDataModuleConfig(BaseConfig, ):
    batch_size: int = 16
    num_workers: int = 11
    val_num_workers: int = 5
    
def worker_init_fn(worker_id, mode):
    pid = os.getpid()
    logger.info(f"Initialized new Worker {worker_id} from {mode} DataLoader with PID {pid}")


class DefaultDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DefaultDataModuleConfig, datasets: list[Dataset]):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        self.data_config = data_config
        # Will be set in setup()
        self.train_dataset = datasets[0]
        self.val_dataset = datasets[1]
        self.test_dataset = datasets[2]

    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        self.train_dataset.prepare_data()
        self.val_dataset.prepare_data()
        self.test_dataset.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit, test, predict).
        """

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
            worker_init_fn=partial(worker_init_fn, mode="train"),
            #pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.val_num_workers,
            shuffle=True,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=partial(worker_init_fn, mode="val"),
            #pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.val_num_workers,
            shuffle=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=partial(worker_init_fn, mode="test"),
            #pin_memory=True,
        )

    def on_exception(self, exception):
        """
        Handle exceptions during training.
        """
        print(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here