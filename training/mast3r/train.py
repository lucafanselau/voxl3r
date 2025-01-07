from argparse import ArgumentParser
import os
from pathlib import Path
import traceback
from typing import Optional, Tuple, Union
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.profilers import AdvancedProfiler
from datasets.transforms.smear_images import SmearMast3r
from networks.u_net import Simple3DUNet, UNet3D, UNet3DConfig
from pydantic import Field
from loguru import logger

import torch
from datasets import chunk, transforms, scene
from networks.voxel_based import VoxelBasedNetworkConfig
from training.mast3r.module_unet3d import UNet3DLightningModule
from training.mast3r.module_voxel_based import VoxelBasedLightningModule
from utils.config import BaseConfig

from training.loggers.occ_grid import OccGridCallback
from training.default.data import DefaultDataModuleConfig, DefaultDataModule
from training.default.module import BaseLightningModule, BaseLightningModuleConfig

class DataConfig(chunk.occupancy_revised.Config, chunk.mast3r.Config, chunk.image.Config, scene.Config, transforms.SmearMast3rConfig, DefaultDataModuleConfig):
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

# class Config(LoggingConfig, VoxelBasedNetworkConfig, BaseLightningModuleConfig, TrainerConfig, DataConfig):
#     resume: Union[bool, str] = False
#     checkpoint_name: str = "last"

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


    RESUME_TRAINING = config.resume != False
    data_config = config

    torch.set_float32_matmul_precision("medium")

    # if we have an identifier, we also want to check if we have a wandb run with that identifier
    if identifier is not None:
        last_ckpt_folder = Path(f"./.lightning/{data_config.name}/{data_config.name}/{identifier}")
        if last_ckpt_folder.exists():
            logger.info(f"Found wandb run {identifier}, resuming training")
            RESUME_TRAINING = True
        else:
            last_ckpt_folder = None
    elif RESUME_TRAINING:
        if not isinstance(config.resume, str):
            ckpt_folder = list(Path(f"./.lightning/{data_config.name}/{data_config.name}/").glob("*"))
            ckpt_folder = sorted(ckpt_folder, key=os.path.getmtime)
            last_ckpt_folder = ckpt_folder[-1]
        else:
            last_ckpt_folder = Path(f"./.lightning/{data_config.name}/{data_config.name}/{config.resume}")
    else:
        last_ckpt_folder = None

    # Setup logging
    if RESUME_TRAINING:
        wandb_logger = WandbLogger(
            project=data_config.name,
            save_dir=f"./.lightning/{data_config.name}",
            id=last_ckpt_folder.stem,
            name=run_name,
            resume="allow",
            group=experiment_name,
        )
    else:
        wandb_logger = WandbLogger(
            project=data_config.name,
            save_dir=f"./.lightning/{data_config.name}",
            id=identifier,
            name=run_name,
            group=experiment_name,
        )
        

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

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
            ["precision", "max"],
            ["recall", "max"],
            #["f1", "max"],
            # ["auroc", "max"],
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
    base_dataset.prepare_data()
    image_dataset = chunk.image.Dataset(data_config, base_dataset)

    # zip = chunk.zip.ZipChunkDataset([
    #     image_dataset,
    #     #chunk.occupancy.Dataset(data_config, base_dataset, image_dataset),
    #     chunk.mast3r.Dataset(data_config, base_dataset, image_dataset),
    # ], transform=transforms.SmearMast3rUsingVoxelizedScene(data_config), base_dataset=base_dataset)
    
    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(data_config, base_dataset, image_dataset),
        chunk.mast3r.Dataset(data_config, base_dataset, image_dataset),
    ], transform=SmearMast3r(config))
    
    
    datamodule = DefaultDataModule(data_config=data_config, dataset=zip)

    # Create configs
    #device_stats = DeviceStatsMonitor(cpu_stats=True)

    # module = VoxelBasedLightningModule(module_config=config) 
    module = UNet3DLightningModule(module_config=config)
    
    wandb_logger.watch(module.model, log=None, log_graph=True)

    # Initialize trainer
    trainer_args = {
        **trainer_kwargs,
        "max_epochs": config.max_epochs if config.limit_epochs is None else config.limit_epochs,
        # "profiler": "simple",
        "log_every_n_steps": config.log_every_n_steps,
        #"callbacks": [*trainer_kwargs.get("callbacks", []), last_callback, *callbacks, voxel_grid_logger, lr_monitor, device_stats],
        "callbacks": [*trainer_kwargs.get("callbacks", []), last_callback, *callbacks, voxel_grid_logger, lr_monitor],
        "logger": wandb_logger,
        "precision": "bf16-mixed", 
        "default_root_dir": "./.lightning/mast3r-3d",
        "limit_val_batches": config.limit_val_batches,
        # overfit settings
        # "overfit_batches": 1,
        # "check_val_every_n_epoch": None,
        # "val_check_interval": 4000,
    }
    
    #profiler = AdvancedProfiler(dirpath="./profiler_logs", filename="perf_logs")
    trainer = Trainer(
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        **trainer_args,
        #profiler=profiler
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
    finally:
        return trainer, module
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
    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
        #"./config/data/undistorted_scenes.yaml"
    ])

    parser = ArgumentParser()

    parser.add_argument("--resume", action="store_true", help="Resume training from *last* checkpoint")
    parser.add_argument("--resume-run", dest="resume_run", type=str, help="Resume training from a specific run")
    parser.add_argument("--ckpt", dest="checkpoint_name", type=str, default="last", help="Name of the checkpoint to resume from")
    args = parser.parse_args()
    logger.info(f"Parsed args: {args}")

    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/module/base.yaml",
    ], {
        **vars(args),
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
        "resume": args.resume_run if args.resume_run is not None else args.resume,
    })
    
    config.max_epochs = 2
    
    config.refinement_blocks = "inceptionBlockA"
    config.name = "mast3r-3d-experiments"

    train({}, config, experiment_name="monitor_memory")


if __name__ == "__main__":
    main()



    

