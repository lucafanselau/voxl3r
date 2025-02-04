from multiprocessing import Manager
from typing import List
from loguru import logger
import torch
import torchvision
from training.default.data import DefaultDataModule, DefaultDataModuleConfig
from datasets import scene, chunk, transforms_batched
from datasets import transforms
import torch.nn as nn

# Load config from checkpoint
def load_config_from_checkpoint(project_name, run_name, checkpoint_name = "last", ConfigClass = None):
    path = f".lightning/{project_name}/{project_name}/{run_name}/checkpoints/{checkpoint_name}.ckpt"
    loaded = torch.load(path)
    data_config = loaded["datamodule_hyper_parameters"]["data_config"]
    if ConfigClass is None:
        return data_config, path
    return ConfigClass(**data_config.model_dump()), path


def create_dataset_rgb(config, split: str, transform=nn.Module):
    config.scenes = None
    config.split = split

    transform.transforms = [transform(config, None) for transform in transform.transforms]
    
    base_dataset = scene.Dataset(config)
    base_dataset.prepare_data()
    image_dataset = chunk.image_loader_compressed.Dataset(config, base_dataset, split=split)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
    ], transform=transform)

    return zip

def create_dataset_rgb_lmdb(config, split: str, transform=nn.Module):
    #config.scenes = None
    config.split = split
    logger.info(f"Creating dataset for split {split}")

    base_dataset = scene.Dataset(config)
    base_dataset.prepare_data()

    image_dataset = chunk.image_loader_lmdb.Dataset(config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(config, base_dataset, image_dataset),
    ], transform=transform)

    return zip
    
def create_dataset(config, split: str, transform=List[nn.Module]):
    #config.scenes = None
    config.split = split
    logger.info(f"Creating dataset for split {split}")
    
    transform.transforms = [transform(config, None) for transform in transform.transforms]

    base_dataset = scene.Dataset(config)
    base_dataset.prepare_data()
    image_dataset = chunk.image.Dataset(config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.mast3r.Dataset(config, base_dataset, image_dataset),
    ], transform=transform)

    return zip

def create_dataset_lmdb(config, split: str, transform=nn.Module):
    #config.scenes = None
    config.split = split
    logger.info(f"Creating dataset for split {split}")

    base_dataset = scene.Dataset(config)
    base_dataset.prepare_data()
    image_dataset = chunk.image.Dataset(config, base_dataset)

    zip = chunk.zip.ZipChunkDataset([
        image_dataset,
        chunk.occupancy_revised.Dataset(config, base_dataset, image_dataset),
        chunk.mast3r_lmdb.Dataset(config, base_dataset, image_dataset),
    ], transform=transform)

    return zip

class DataConfig(chunk.occupancy_revised.Config, DefaultDataModuleConfig, transforms.ComposeTransformConfig):
    pass

class DataConfigRGB(DataConfig, chunk.image_loader.Config, transforms.SmearImagesConfig):
    pass



def create_datamodule_rgb(config: DataConfigRGB, splits = ["train", "val", "test"], DataModuleClass = DefaultDataModule):

    datasets = { split: create_dataset_rgb(config, split, transform=transforms.ComposeTransforms(config)) for split in splits }
    
    collate_fns = {}
    for split in splits:
        config.split = split
        collate_fn = transforms_batched.ComposeTransforms(config)
        collate_fn.transforms = [transform(config, None) for transform in collate_fn.transforms]
        collate_fns[split] = collate_fn
        
    datamodule = DataModuleClass(data_config=config, datasets=datasets, collate_fn=collate_fns)
    return datamodule

class DataConfigRGBLMDB(DataConfig, chunk.image_loader_lmdb.Config, transforms.SmearImagesConfig):
    pass

def create_datamodule_rgb_lmdb(config: DataConfigRGBLMDB, splits = ["train", "val", "test"], DataModuleClass = DefaultDataModule, collate_fn=None):
    transform = transforms.ComposeTransforms(config)
    datasets = { split: create_dataset_rgb_lmdb(config, split, transform=transform) for split in splits }
    datamodule = DataModuleClass(data_config=config, datasets=datasets, collate_fn=collate_fn)
    return datamodule

class DataConfigMast3r(DataConfig, chunk.mast3r.Config, transforms.SmearMast3rConfig):
    pass

def create_datamodule(config: DataConfigMast3r, splits = ["train", "val", "test"], DataModuleClass = DefaultDataModule, collate_fn=None):

    datasets = { split: create_dataset(config, split, transform=transforms.ComposeTransforms(config)) for split in splits }
    
    collate_fns = {}
    for split in splits:
        config.split = split
        collate_fn = transforms_batched.ComposeTransforms(config)
        collate_fn.transforms = [transform(config, None) for transform in collate_fn.transforms]
        collate_fns[split] = collate_fn

    datamodule = DataModuleClass(data_config=config, datasets=datasets, collate_fn=collate_fns)
    return datamodule

class DataConfigLMDB(DataConfig, chunk.mast3r_lmdb.Config, transforms.SmearMast3rConfig):
    pass

def create_datamodule_lmdb(config: DataConfigLMDB, splits = ["train", "val", "test"], DataModuleClass = DefaultDataModule, collate_fn=None):
    transform = transforms.ComposeTransforms(config)
    datasets = { split: create_dataset_lmdb(config, split, transform=transform) for split in splits }
    datamodule = DataModuleClass(data_config=config, datasets=datasets, collate_fn=collate_fn)
    return datamodule
