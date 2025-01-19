

from lightning import Trainer, Callback
from loguru import logger
import torch
from datasets.transforms.smear_images import SmearMast3r
from training.default.data import DefaultDataModule
from training.mast3r.module_transformer_unet3D import TransformerUNet3DLightningModule
from training.mast3r.module_unet3d import UNet3DLightningModule
from training.mast3r.train_transformer import Config as Mast3rConfig
from training.mast3r.train import Config as UNet3DConfig
from lightning.pytorch.loggers import WandbLogger

from training.common import load_config_from_checkpoint, prepare_datasets


run_name = "K9YT7_1_0"
group = None
project_name = "mast3r-3d-experiments"
DataModule = DefaultDataModule
Module = UNet3DLightningModule
ConfigClass = UNet3DConfig


def eval_run(run_name, project_name):

    config, path = load_config_from_checkpoint(project_name, run_name, ConfigClass=ConfigClass)
    datamodule = prepare_datasets(config, splits=["test"])

    # custom migrations
    if hasattr(config, "disable_batchnorm"):
        config.disable_norm = config.disable_batchnorm


    module = Module.load_from_checkpoint(path, module_config=config)

    wandb_logger = WandbLogger(
        project=config.name,
        save_dir=f"./.lightning/{config.name}",
        group=group,
        #id=run_name,
        #resume="allow",
    )


    trainer = Trainer(max_epochs=1, logger=wandb_logger)

    logger.info("Starting evaluation")
    trainer.test(module, datamodule)
    logger.info("Evaluation finished")
    

if __name__ == "__main__":
    eval_run(run_name, project_name)