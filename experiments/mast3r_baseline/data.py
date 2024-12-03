# Transform to create

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from experiments.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
import lightning.pytorch as pl

from torch.utils.data import DataLoader, random_split

from extern.mast3r.dust3r.dust3r.utils.image import load_images
from utils.data_parsing import load_yaml_munch


config = load_yaml_munch("./utils/config.yaml")

@dataclass
class Mast3rBaselineDataTransformConfig:
    # the default here is the same as in their demo
    image_size: int = 512


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

        occ, data_dict = data

        image_names, transforms = data_dict["images"]
        
        image_names = [str(Path(config.data_dir) / Path(*Path(image_name).parts[Path(image_name).parts.index("data") + 3 :])) for image_name in image_names]

        images = load_images(image_names, self.config.image_size, verbose = False)

        return occ, images, image_names, transforms, {"center" : data_dict["center"], "resolution" : data_dict["resolution"], "grid_size" : data_dict["grid_size"]}


@dataclass
class Mast3rBaselineDataConfig(
    OccChunkDatasetConfig, Mast3rBaselineDataTransformConfig
):
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
