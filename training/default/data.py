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
    prefetch_factor: int = 2
    
def worker_init_fn(worker_id, mode):
    pid = os.getpid()
    logger.info(f"Initialized new Worker {worker_id} from {mode} DataLoader with PID {pid}")


class DefaultDataModule(pl.LightningDataModule):
    def __init__(self, data_config: DefaultDataModuleConfig, datasets: dict[str, Dataset], collate_fn=None):
        super().__init__()
        self.save_hyperparameters(ignore=["datasets", "collate_fn"])
        self.data_config = data_config
        # Will be set in setup()
        self.train_dataset = datasets.get("train", None)
        self.val_dataset = datasets.get("val", None)
        self.test_dataset = datasets.get("test", None)
        
        if isinstance(collate_fn, dict):
            self.train_collate_fn = collate_fn.get("train", None)
            self.val_collate_fn = collate_fn.get("val", None)
            self.test_collate_fn = collate_fn.get("test", None)
        else:
            self.train_collate_fn = collate_fn
            self.val_collate_fn = collate_fn
            self.test_collate_fn = collate_fn
            
    def prepare_data(self):
        """
        Download or prepare data. Called only on one GPU.
        """
        if self.train_dataset is not None:
            self.train_dataset.prepare_data()
        if self.val_dataset is not None:
            self.val_dataset.prepare_data()
        if self.test_dataset is not None:
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
        if self.train_dataset is None:
            raise ValueError("Train dataset was not provided to datamodule")

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            shuffle=True,
            prefetch_factor=self.data_config.prefetch_factor if self.data_config.num_workers > 0 else None,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=partial(worker_init_fn, mode="train"),
            collate_fn=self.train_collate_fn,
            #pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Validation dataset was not provided to datamodule")
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.val_num_workers,
            shuffle=True,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            prefetch_factor=self.data_config.prefetch_factor if self.data_config.num_workers > 0 else None,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=partial(worker_init_fn, mode="val"),
            collate_fn=self.val_collate_fn,
            #pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset was not provided to datamodule")
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.val_num_workers,
            shuffle=False,
            persistent_workers=True if self.data_config.num_workers > 0 else False,
            generator=torch.Generator().manual_seed(42),
            worker_init_fn=partial(worker_init_fn, mode="test"),
            collate_fn=self.test_collate_fn,
            #pin_memory=True,
        )

    def on_exception(self, exception):
        """
        Handle exceptions during training.
        """
        print(f"Exception occurred: {exception}")
        # Add any additional cleanup or logging here