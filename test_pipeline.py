from argparse import ArgumentParser
import os
from pathlib import Path
import traceback
from typing import Optional, Tuple, Union
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.profilers import AdvancedProfiler
import torchvision
from networks.surfacenet import SurfaceNet
from networks.u_net import  UNet3D, UNet3DConfig
from pydantic import Field
from loguru import logger

import torch
from datasets import chunk, transforms, transforms_batched
from training.common import create_datamodule_rgb
from utils.config import BaseConfig

from training.loggers.occ_grid import OccGridCallback
from training.default.data import DefaultDataModuleConfig, DefaultDataModule
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

class DataConfig(chunk.occupancy_revised.Config, chunk.image_loader_compressed.Config, transforms.SmearImagesConfig, DefaultDataModuleConfig, transforms.ComposeTransformConfig, transforms_batched.ComposeTransformConfig, transforms_batched.SampleOccGridConfig):
    name: str = "mast3r-3d"

class LoggingConfig(BaseConfig):
    grid_occ_interval: Tuple[int, int, int] = Field(default=(4, 4, 1))
    save_top_k: int = 3
    log_every_n_steps: int = 1

class TrainerConfig(BaseConfig):
    max_epochs: int = 300
    # overrides only the max_epochs for the trainer, not the max_epochs for the lr_scheduler and tuner
    limit_epochs: Optional[int] = None
    limit_val_batches: int = 16
    check_val_every_n_epoch: int = 1


class Config(LoggingConfig, UNet3DConfig, BaseLightningModuleConfig, TrainerConfig, DataConfig):
    resume: Union[bool, str] = False
    checkpoint_name: str = "last"


def train(
        config: dict, 
        default_config: Config, 
        trainer_kwargs: dict = {}, 
        identifier: Optional[str] = None,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):

    config: Config = Config(**{**default_config.model_dump(), **config}) 
    logger.debug(f"Config: {config}")


    datamodule = create_datamodule_rgb(config, splits=["train"])
    datamodule.prepare_data()

    dataloader = datamodule.train_dataloader()
    for batch in dataloader:
        pass

def main():
    # first load data_config
    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
        # "./config/data/images_transform.yaml",
        "./config/data/images_transform_batched.yaml",
    ])
    
    #data_config.add_confidences = True
    #data_config.add_pts3d = True    

    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/module/base.yaml",
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
    })
    
    
    config.disable_norm = False
    config.base_channels = 16
    config.name = "mast3r-3d-experiments"
    config.max_epochs = 100
    config.prefetch_factor = 2
    config.num_workers = 0
    config.val_num_workers = 0
    config.check_val_every_n_epoch = 1
    
    config.num_refinement_blocks = 3
    config.refinement_bottleneck = 6
    config.refinement_blocks = "simpleWithSkip"

    config.scenes = ["fd361ab85f"]
 
    #config.force_prepare_mast3r = True
    #config.force_prepare = True

    train({}, config, experiment_name="monitor_memory")


if __name__ == "__main__":
    main()



    

