

from lightning import Trainer, Callback
from loguru import logger
import torch
from datasets.transforms.smear_images import SmearMast3r
from training.default.data import DefaultDataModule
from training.mast3r.module_transformer_unet3D import TransformerUNet3DLightningModule
from training.mast3r.module_unet3d import UNet3DLightningModule
from training.mast3r.train_transformer import Config as Mast3rConfig
from training.mast3r.train import Config as UNet3DConfig
from datasets import chunk, transforms, scene
from lightning.pytorch.loggers import WandbLogger


run_name = "K9YT7_1_0"
group = None
project_name = "mast3r-3d-experiments"
DataModule = DefaultDataModule
Module = UNet3DLightningModule
ConfigClass = UNet3DConfig


def eval_run(run_name, project_name):

    path = f".lightning/{project_name}/{project_name}/{run_name}/checkpoints/last.ckpt"

    loaded = torch.load(path)
    data_config = loaded["datamodule_hyper_parameters"]["data_config"]

    data_config = ConfigClass(**data_config.model_dump())
    config = data_config

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
    ], transform=SmearMast3r(data_config))
    
    
    datamodule = DefaultDataModule(data_config=data_config, dataset=zip)

    # custom map stuff (this is only needed to compat)
    # config = loaded["hyper_parameters"]["module_config"]

    if hasattr(config, "disable_batchnorm"):
        config.disable_norm = config.disable_batchnorm


    module = Module.load_from_checkpoint(path, module_config=config)

    wandb_logger = WandbLogger(
        project=data_config.name,
        save_dir=f"./.lightning/{data_config.name}",
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