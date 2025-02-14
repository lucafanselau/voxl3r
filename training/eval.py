from lightning import Trainer, Callback
from loguru import logger
import torch
from networks.attention_net import AttentionNet
from networks.smear_module import SmearImages
from networks.mast3r_module import Mast3rModule
from networks.patch_2_pixels import Patch2Pixels
from networks.surfacenet import SurfaceNet
from training.default.aux_module import LightningModuleWithAux
from training.default.data import DefaultDataModule
from training.default.module import BaseLightningModule
from training.loggers.occ_grid import OccGridCallback

# from training.mast3r.module_transformer_unet3D import TransformerUNet3DLightningModule
# from training.mast3r.module_unet3d import UNet3DLightningModule
# from training.mast3r.train_transformer import Config as Mast3rConfig
from training.mast3r.train_rgb import Config as TrainConfigRGB
from training.mast3r.train_attention import Config as TrainConfig
from lightning.pytorch.loggers import WandbLogger

from training.common import (
    load_config_from_checkpoint,
    create_datamodule,
    create_datamodule_rgb,
)


run_name = "1tdd32ce"  # BEST local feat "ohkmg3nr"  # BEST feat based "wxklqj28"
# group = "08_trial_transformer_unet3d"
project_name = "mast3r-3d-experiments"
DataModule = DefaultDataModule
Module = LightningModuleWithAux  # BaseLightningModule #UNet3DLightningModule
ConfigClass = TrainConfig  # UNet3DConfig
Config = ConfigClass


def eval_run(run_name, project_name):

    config, path = load_config_from_checkpoint(
        project_name, run_name, ConfigClass=ConfigClass
    )
    datamodule = create_datamodule(config, splits=["test"])

    # custom migrations
    if hasattr(config, "disable_batchnorm"):
        config.disable_norm = config.disable_batchnorm

    module = Module.load_from_checkpoint(
        path, module_config=config, ModelClass=AttentionNet
    )

    wandb_logger = WandbLogger(
        project=config.name,
        save_dir=f"./.lightning/{config.name}",
        # group=group,
        # id=run_name,
        # resume="allow",
    )

    occ_grid_callback = OccGridCallback(wandb_logger, n_epochs=(1, 1, 1), max_results=8)

    # datamodule.prepare_data()
    # test_dataloader = datamodule.test_dataloader()

    # batch = next(iter(test_dataloader))

    trainer = Trainer(
        max_epochs=1,
        logger=wandb_logger,
        callbacks=[occ_grid_callback],
    )

    # outputs = {
    #     "pred": torch.randn(8, 1, 1, 32, 32, 32),
    #     "loss_batch": torch.randn(8, 1),
    # }
    # occ_grid_callback.on_test_batch_end(trainer, module, outputs, batch, 0)

    logger.info("Starting evaluation")
    trainer.test(module, datamodule)
    logger.info("Evaluation finished")

    # try to close occ_grid_callback
    del occ_grid_callback


if __name__ == "__main__":
    eval_run(run_name, project_name)
