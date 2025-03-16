from argparse import ArgumentParser
import os
from pathlib import Path
import traceback
from typing import Optional, Tuple, Union
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from lightning.pytorch.profilers import AdvancedProfiler
from datasets.transforms.sample_coordinate_grid import SampleCoordinateGridConfig
from networks.aggregator_net import AggregatorNet
from networks.aggregator_net_with_downsampling import AggregatorNetWithDownsampling
from networks.attention_net import AttentionNet, AttentionNetConfig
from networks.surfacenet_with_attention import SurfaceNetTransformer
from networks import surfacenet_with_attention
from networks.surfacenet_with_downscaling import SurfaceNetWithDownscaling
from networks.u_net import UNet3DConfig
from pydantic import Field
from loguru import logger

import torch
from datasets import chunk, transforms, scene, transforms_batched
from training.common import create_datamodule
from training.default.aux_module import (
    LightningModuleWithAux,
    LightningModuleWithAuxConfig,
)
from utils.config import BaseConfig

from training.loggers.occ_grid import OccGridCallback
from training.default.data import DefaultDataModuleConfig, DefaultDataModule
from training.default.module import BaseLightningModule, BaseLightningModuleConfig


class DataConfig(
    chunk.mast3r.Config,
    chunk.occupancy_revised.Config,
    chunk.image_loader_compressed.Config,
    transforms.SmearMast3rConfig,
    DefaultDataModuleConfig,
    transforms.ComposeTransformConfig,
    transforms_batched.ComposeTransformConfig,
    transforms_batched.SampleOccGridConfig,
    SampleCoordinateGridConfig,
):
    name: str = "mast3r-3d"


class LoggingConfig(BaseConfig):
    grid_occ_interval: Tuple[int, int, int] = Field(default=(3, 3, 1))
    save_top_k: int = 1
    log_every_n_steps: int = 1


class TrainerConfig(BaseConfig):
    max_epochs: int = 300
    # overrides only the max_epochs for the trainer, not the max_epochs for the lr_scheduler and tuner
    limit_epochs: Optional[int] = None
    limit_val_batches: Optional[int] = None
    accumulate_grad_batches: Optional[int] = 1
    check_val_every_n_epoch: int = 1


# ModelClass = [SurfaceNetWithDownscaling, AggregatorNetWithDownsampling] #AttentionNet  # UNet
ModelClass = SurfaceNetTransformer # AttentionNet  # UNet
ModelClassConfig = surfacenet_with_attention.SmallSurfaceNetConfig


class Config(
    LoggingConfig,
    ModelClassConfig,
    LightningModuleWithAuxConfig,
    TrainerConfig,
    DataConfig,
    AttentionNetConfig,
):
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
        last_ckpt_folder = Path(
            f"./.lightning/{data_config.name}/{data_config.name}/{identifier}"
        )
        if last_ckpt_folder.exists():
            logger.info(f"Found wandb run {identifier}, resuming training")
            RESUME_TRAINING = True
        else:
            last_ckpt_folder = None
    elif RESUME_TRAINING:
        if not isinstance(config.resume, str):
            ckpt_folder = list(
                Path(f"./.lightning/{data_config.name}/{data_config.name}/").glob("*")
            )
            ckpt_folder = sorted(ckpt_folder, key=os.path.getmtime)
            last_ckpt_folder = ckpt_folder[-1]
        else:
            last_ckpt_folder = Path(
                f"./.lightning/{data_config.name}/{data_config.name}/{config.resume}"
            )
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
        # for type in ["train", "val"]
        # for [name, mode] in [
        #     ["loss", "min"],
        #     ["accuracy", "max"],
        #     ["precision", "max"],
        #     ["recall", "max"],
        #     # ["f1", "max"],
        #     # ["auroc", "max"],
        # ]
        for type in ["val"]
        for [name, mode] in [
            ["loss", "min"],
            # ["f1", "max"],
            # ["auroc", "max"],
        ]
    ]
    # Save the model every 5 epochs
    last_callback = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=1,
        save_last=True,
    )

    # TODO: Fix before the next run
    last_callback = ModelCheckpoint(
        every_n_epochs=2,
        save_top_k=-1,
    )
    # Custom callback for logging the 3D voxel grids
    voxel_grid_logger = OccGridCallback(
        wandb=wandb_logger, n_epochs=config.grid_occ_interval
    )

    datamodule = create_datamodule(config, splits=["train", "val"])

    # Create configs
    # device_stats = DeviceStatsMonitor(cpu_stats=True)

    # module = VoxelBasedLightningModule(module_config=config)
    # module = LightningModuleWithAux(module_config=config, ModelClass=AggregatorNet)
    samplerConfig = config.model_copy()
    samplerConfig.split = None
    occGridSampler = transforms_batched.ComposeTransforms(samplerConfig)
    module = BaseLightningModule(
        config=config, ModelClass=ModelClass, occGridSampler=occGridSampler
    )

    # module = LightningModuleWithAux(module_config=config, ModelClass=AttentionNet, occGridSampler=occGridSampler)

    wandb_logger.watch(module.model, log=None, log_graph=True)

    # Initialize trainer
    trainer_args = {
        **trainer_kwargs,
        "max_epochs": (
            config.max_epochs if config.limit_epochs is None else config.limit_epochs
        ),
        # "profiler": "simple",
        "log_every_n_steps": config.log_every_n_steps,
        # "callbacks": [*trainer_kwargs.get("callbacks", []), last_callback, *callbacks, voxel_grid_logger, lr_monitor, device_stats],
        "callbacks": [
            *trainer_kwargs.get("callbacks", []),
            last_callback,
            *callbacks,
            lr_monitor,
            # voxel_grid_logger,
        ],
        "logger": wandb_logger,
        "precision": "bf16-mixed",
        "default_root_dir": "./.lightning/mast3r-3d",
        "limit_val_batches": config.limit_val_batches,
        # "limit_train_batches": 10,  # config.limit_train_batches,
        # overfit settings
        # "overfit_batches": 1,
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        # "val_check_interval": 4000,
        "num_sanity_val_steps": 0,
    }

    # profiler = AdvancedProfiler(dirpath="./profiler_logs", filename="perf_logs")
    trainer = Trainer(
        **trainer_args,
        gradient_clip_val=1.0,
        # profiler=profiler
    )

    if RESUME_TRAINING:
        logger.info(f"Resuming training from {last_ckpt_folder}")

    try:
        trainer.fit(
            module,
            datamodule=datamodule,
            ckpt_path=(
                last_ckpt_folder / f"checkpoints/{config.checkpoint_name}.ckpt"
                if RESUME_TRAINING
                else None
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
    data_config = DataConfig.load_from_files(
        [
            "./config/data/base.yaml",
            "./config/data/mast3r.yaml",
            "./config/data/mast3r_transform.yaml",
            "./config/data/mast3r_transform_batched.yaml",
        ]
    )

    # data_config.add_confidences = True
    # data_config.add_pts3d = True

    parser = ArgumentParser()

    parser.add_argument(
        "--resume", action="store_true", help="Resume training from *last* checkpoint"
    )
    parser.add_argument(
        "--resume-run",
        dest="resume_run",
        type=str,
        help="Resume training from a specific run",
    )
    parser.add_argument(
        "--ckpt",
        dest="checkpoint_name",
        type=str,
        default="last",
        help="Name of the checkpoint to resume from",
    )
    args = parser.parse_args()
    logger.info(f"Parsed args: {args}")

    config = Config.load_from_files(
        [
            "./config/trainer/base.yaml",
            "./config/network/base_unet.yaml",
            # "./config/network/unet3D.yaml",
            "./config/network/aux_network.yaml",
            "./config/module/base.yaml",
        ],
        {
            **vars(args),
            **data_config.model_dump(),
            "in_channels": data_config.get_feature_channels(),
            "resume": args.resume_run if args.resume_run is not None else args.resume,
        },
    )

    config.scenes = [
        path.name
        for path in Path("/mnt/dorta/scannetpp/preprocessed").iterdir()
        if path.is_dir()
    ]

    config.name = "mast3r-3d-experiments"
    config.max_epochs = 100
    config.num_workers = 11
    config.val_num_workers = 4
    config.target_chunks = None
    config.use_masked_loss = True

    # config.grid_resolution_sample_fine = 0.01
    # config.grid_size_sample_fine = [64, 64, 64]

    config.use_abs_pe = True
    config.use_aux_loss = False

    config.enable_rotation = False
    
    # config transformer
    config.per_voxel_channels = 64
    config.dim = 128
    config.depth = 6
    config.heads = 4
    config.mlp_dim = 4*128
    config.dim_head  = 32
    
    config.requires_grad = True
    
    # config schedular
    config.scheduler = "LinearWarmupCosineAnnealingLR"
    config.eta_min = 0.0

    # config.force_prepare_mast3r = True
    # config.mast3r_keys = ["desc"]

    # config.force_prepare_mast3r = True
    # config.force_prepare = True

    train({}, config, experiment_name="monitor_memory")


if __name__ == "__main__":
    main()
