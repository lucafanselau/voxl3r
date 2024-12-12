from argparse import ArgumentParser
import os
from pathlib import Path
import traceback
from typing import Any, Tuple, Union
from lightning import Trainer
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from networks.u_net import Simple3DUNetConfig, Simple3DUNet
from pydantic import Field
from loguru import logger

import torch
from datasets import chunk, transforms, scene
from utils.config import BaseConfig

from training.loggers.occ_grid import OccGridCallback
from training.default.data import DefaultDataModuleConfig, DefaultDataModule
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

class Mast3R3DDataConfig(chunk.occupancy.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig, DefaultDataModuleConfig):
    name: str = "mast3r-3d"

class LoggingConfig(BaseConfig):
    grid_occ_interval: Tuple[int, int, int] = Field(default=(4, 4, 1))
    save_top_k: int = 3
    log_every_n_steps: int = 1

class TrainerConfig(BaseConfig):
    max_epochs: int = 300
    limit_val_batches: int = 16
    check_val_every_n_epoch: int = 1


class Config(LoggingConfig, Simple3DUNetConfig, BaseLightningModuleConfig, TrainerConfig, Mast3R3DDataConfig):
    resume: Union[bool, str] = False
    checkpoint_name: str = "last"

def train(config, default_config: Config, trainer_kwargs: dict = {}):

    config: Config = Config(**{k: config[k] | v if k in config else v for k, v in default_config.model_dump().items()}) 

    RESUME_TRAINING = config.resume != False
    data_config = config

    torch.set_float32_matmul_precision("medium")

    # load ckpt
    if RESUME_TRAINING:
        ckpt_folder = list(Path(f"./.lightning/{data_config.name}/{data_config.name}/").glob("*"))
        ckpt_folder = sorted(ckpt_folder, key=os.path.getmtime)
        last_ckpt_folder = ckpt_folder[-1]

    # Setup logging
    if RESUME_TRAINING:
        wandb_logger = WandbLogger(
            project=data_config.name,
            save_dir=f"./.lightning/{data_config.name}",
            id=last_ckpt_folder.stem,
            resume="allow",
        )
    else:
        wandb_logger = WandbLogger(
            project=data_config.name,
            save_dir=f"./.lightning/{data_config.name}",
        )

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Checkpointing callbacks
    filename = "{epoch}-{step}"
    callbacks = [
        ModelCheckpoint(
            filename=filename + f"-{{{type}_{name}:.2f}}",
            monitor=f"{type}_{name}",
            save_top_k=config.save_top_k,
            mode=mode,
        )
        for type in ["train", "val"]
        for [name, mode] in [
            ["loss", "min"],
            ["accuracy", "max"],
            ["f1", "max"],
            ["auroc", "max"],
        ]
    ]
    # Save the model every 5 epochs
    last_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
    )

    # Custom callback for logging the 3D voxel grids
    voxel_grid_logger = OccGridCallback(wandb=wandb_logger, n_epochs=config.grid_occ_interval)

    # Train
    base_dataset = scene.Dataset(data_config)
    image_dataset = chunk.image.Dataset(data_config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy.Dataset(data_config, base_dataset, image_dataset),
        chunk.mast3r.Dataset(data_config, base_dataset, image_dataset),
    ], transform=transforms.SmearMast3r(data_config))
    datamodule = DefaultDataModule(data_config=data_config, dataset=zip)

    # Create configs
    device_stats = DeviceStatsMonitor()

    module = BaseLightningModule(module_config=config, ModelClass=Simple3DUNet, model_config=config)

    # Initialize trainer
    trainer_args = {
        **trainer_kwargs,
        "max_epochs": config.max_epochs,
        # profiler="simple",
        "log_every_n_steps": config.log_every_n_steps,
        "callbacks": [*trainer_kwargs.get("callbacks", []), last_callback, *callbacks, lr_monitor, voxel_grid_logger, device_stats],
        "logger": wandb_logger,
        "precision": "bf16-mixed", 
        "default_root_dir": "./.lightning/mast3r-3d",
        "limit_val_batches": config.limit_val_batches,
        # overfit settings
        # "overfit_batches": 1,
        # "check_val_every_n_epoch": None,
        # "val_check_interval": 4000,
    }
    # profiler = "advanced" # PyTorchProfiler()
    trainer = Trainer(
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        **trainer_args,
        # profiler=profiler
    )

    if RESUME_TRAINING:
        logger.info(f"Resuming training from {last_ckpt_folder}")

    try:
        trainer.fit(
            module,
            datamodule=datamodule,
            ckpt_path=(
                last_ckpt_folder / f"checkpoints/{config.checkpoint_name}.ckpt" if RESUME_TRAINING else None
            ),
        )
    except Exception as e:
        logger.error(f"Training failed with error, finishing training")
        print(traceback.format_exc())
    # finally:
        # Save best checkpoints info
        # base_path = Path(callbacks[0].best_model_path).parents[1]
        # result = base_path / "best_ckpts.pt"
        # result_dict = {
        #     f"best_model_{type}_{name}": callbacks[i * 4 + j].best_model_path
        #     for i, type in enumerate(["train", "val"])
        #     for j, [name, mode] in enumerate([
        #         ["loss", "min"],
        #         ["accuracy", "max"],
        #         ["f1", "max"],
        #         ["auroc", "max"],
        #     ])
        # }
        # result_dict["last_model_path"] = every_five_epochs.best_model_path
        # torch.save(result_dict, result)
        # return result_dict, module, trainer, datamodule


def main():
    # first load data_config
    data_config = Mast3R3DDataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/mast3r_scenes.yaml"
    ])

    parser = ArgumentParser()

    parser.add_argument("--resume", action="store_true", help="Resume training from *last* checkpoint")
    parser.add_argument("--ckpt", dest="checkpoint_name", type=str, default="last", help="Name of the checkpoint to resume from")
    args = parser.parse_args()
    logger.info(f"Parsed args: {args}")

    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/network/base_unet.yaml",
        "./config/module/base.yaml"
    ], {
        **vars(args),
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
    })

    train({}, config)


if __name__ == "__main__":
    main()



    

