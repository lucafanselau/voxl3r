from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from torch.utils.data import Dataset
from multiprocessing import Pool
import torch
import tqdm

from datasets.scene import Dataset
from utils.config import BaseConfig


class ChunkBaseDatasetConfig(BaseConfig):
    data_dir: str
    camera: str

    storage_preprocessing: str = "preprocessed"

    scenes: Optional[List[str]] = None
    num_workers: int = 1
    force_prepare: bool = False

    overfit_mode: bool = False
    skip_prepare: bool = False


class ChunkBaseDataset(Dataset, ABC):
    """
    Defines a base class for chunk datasets.

    Chunk datasets, take the scenes and split them up into chunks, where a chunk is identified by a set of images.

    The chunks are stored in the data_dir, and are identified by a file name of the first image in the chunk.

    The convention is that the first image also defines to coordinate system of the chunk.
    """

    def __init__(self, data_config: ChunkBaseDatasetConfig, base_dataset: Dataset):
        self.data_config = data_config
        self.base_dataset = base_dataset
        self.file_names: Optional[dict[str, List[Path]]] = None
        self.cache = None
        self.prepared = False

    @abstractmethod
    def get_chunk_dir(self, scene_name: str) -> Path:
        pass

    # Altough this is not abstract, it is required for the chunking to work
    # If not present a custom implementation for prepare_data is required
    def create_chunks_of_scene(self, base_dataset_dict: dict):
        pass

    @abstractmethod
    def get_at_idx(self, idx: int, fallback: bool = False):
        pass

    def check_chunks_exists(self, scene_name: str):
        raise NotImplementedError("This method is not implemented for the Base Dataset")

    def get_identifiers(
        self, scenes: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        if scenes is None:
            # return all file names
            return [
                (scene, file.name)
                for scene, files in self.file_names.items()
                for file in files
            ]
        else:
            return [
                (scene, file.name)
                for scene, files in self.file_names.items()
                if scene in scenes
                for file in files
            ]

    @abstractmethod
    def get_at_idx(self, idx: int, fallback: bool = False):
        pass

    def __getitem__(self, idx):
        if not self.prepared and self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )
        if (
            self.data_config.overfit_mode
            and self.cache is not None
            and idx in self.cache
        ):
            return self.cache[idx]

        result = self.get_at_idx(idx)

        if self.data_config.overfit_mode:
            if self.cache is None:
                self.cache = {}
            self.cache[idx] = result

        return result

    def __len__(self):
        if not self.prepared and self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )
        return sum([len(self.file_names[scene_name]) for scene_name in self.file_names])


    def get_saving_path(self, scene_name: str) -> Path:
        return (
            Path(self.data_config.data_dir)
            / self.data_config.storage_preprocessing
            / scene_name
            / self.data_config.camera
        )

    def on_after_prepare(self):
        pass
        
    def prepare_scene(self, scene_name: str, force: bool = False):

        data_dir = self.get_chunk_dir(scene_name)

        if self.check_chunks_exists(scene_name) and not self.data_config.force_prepare and not force:
            # we need to check if all of the chunks for this scene are present
            logger.trace(f"Chunks for scene {scene_name} already exist. Skipping.")
            return

        idx = self.base_dataset.get_index_from_scene(scene_name)
        mesh_path = (
            self.base_dataset.data_dir
            / self.base_dataset.scenes[idx]
            / "scans"
            / "mesh_aligned_0.05.ply"
        )
        if not mesh_path.exists():
            print(f"Mesh not found for scene {scene_name}. Skipping.")
            return

        i = 0
        for scene_dict, chunk_identifier in self.create_chunks_of_scene(
            self.base_dataset[idx]
        ):
            if not data_dir.exists():
                data_dir.mkdir(parents=True)

            if chunk_identifier is None:
                # image_name = scene_dict["image_name_chunk"]
                # file_name = f"{i}_{image_name}.pt"
                raise ValueError("Chunk identifier is None")
            else:
                file_name = chunk_identifier

            scene_dict["file_name"] = file_name
            torch.save(scene_dict, data_dir / file_name)
            i = i + 1

    def prepare_data(self):
        if self.data_config.skip_prepare:
            self.load_paths()
            self.on_after_prepare()
            self.prepared = True
            return
        scenes = (
            self.data_config.scenes
            if self.data_config.scenes is not None
            else self.base_dataset.scenes
        )
        if self.data_config.num_workers > 1:
            with Pool(self.data_config.num_workers) as p:
                with tqdm.tqdm(total=len(scenes), position=0, leave=True) as pbar:
                    for _ in p.imap_unordered(self.prepare_scene, scenes):
                        pbar.update()
        else:
            for scene_name in tqdm.tqdm(scenes, leave=False):
                self.prepare_scene(scene_name)
        self.load_paths()
        self.on_after_prepare()
        self.prepared = True

    def load_paths(self):
        self.file_names = {}
        scenes = (
            self.data_config.scenes
            if self.data_config.scenes is not None
            else self.base_dataset.scenes
        )
        for scene_name in scenes:
            data_dir = self.get_chunk_dir(scene_name)
            if data_dir.exists():
                self.file_names[scene_name] = [
                    s for s in data_dir.iterdir() if s.is_file()
                ]
