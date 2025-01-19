from loguru import logger
import torch
from training.default.data import DefaultDataModule
from datasets import scene, chunk
import datasets.transforms.smear_images as transforms
import torch.nn as nn

# Load config from checkpoint
def load_config_from_checkpoint(project_name, run_name, checkpoint_name = "last", ConfigClass = None):
    path = f".lightning/{project_name}/{project_name}/{run_name}/checkpoints/{checkpoint_name}.ckpt"
    loaded = torch.load(path)
    data_config = loaded["datamodule_hyper_parameters"]["data_config"]
    if ConfigClass is None:
        return data_config, path
    return ConfigClass(**data_config.model_dump()), path



def create_dataset(config, split: str, transform=nn.Module):
    config.scenes = None
    config.split = split
    logger.info(f"Creating dataset for split {split}")

    base_dataset = scene.Dataset(config)
    base_dataset.prepare_data()
    image_dataset = chunk.image.Dataset(config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(config, base_dataset, image_dataset),
        chunk.mast3r.Dataset(config, base_dataset, image_dataset),
    ], transform=transform(config))

    return zip


def prepare_datasets(config, splits = ["train", "val", "test"], DataModuleClass = DefaultDataModule, transform=transforms.SmearMast3r, collate_fn=None):
    datasets = { split: create_dataset(config, split, transform=transform) for split in splits }

    datamodule = DataModuleClass(data_config=config, datasets=datasets, collate_fn=collate_fn)
    return datamodule
