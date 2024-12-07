from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import Dataset
from multiprocessing import Pool
import torch

from dataset import SceneDataset, SceneDatasetConfig


class ChunkDataset(Dataset, ABC):
    def __init__(
        self,
        data_config: SceneDatasetConfig,
        base_dataset: Optional[SceneDataset] = None,
    ):
        self.data_config = data_config
        self.base_dataset = base_dataset if base_dataset is not None else SceneDataset(data_config)
        self.file_names = None

    @abstractmethod
    def get_chunk_dir(self, scene_name: str) -> Path:
        pass

    @abstractmethod
    def create_chunks_of_scene(self, base_dataset_dict: dict):
        pass

    @abstractmethod
    def get_at_idx(self, idx: int, fallback: bool = False):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    def get_saving_path(self, scene_name: str) -> Path:
        return Path(self.data_config.data_dir) / self.data_config.storage_preprocessing / scene_name / self.data_config.camera

    def prepare_scene(self, scene_name: str):
        data_dir = self.get_chunk_dir(scene_name)
        if data_dir.exists() and not self.data_config.force_prepare:
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
        for scene_dict, chunk_identifier in self.create_chunks_of_scene(self.base_dataset[idx]):
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
            
            if chunk_identifier is None:
                image_name = scene_dict["image_name_chunk"]
                file_name = f"{i}_{image_name}.pt"
            else:
                file_name = chunk_identifier
                
            scene_dict["file_name"] = file_name
            torch.save(scene_dict, data_dir / file_name)
            i = i + 1
            
    def prepare_data(self):
        scenes = self.data_config.scenes if self.data_config.scenes is not None else self.base_dataset.scenes
        if self.data_config.num_workers > 1:
            with Pool(self.data_config.num_workers) as p:
                p.map(self.prepare_scene, scenes)
        else:
            for scene_name in scenes:
                self.prepare_scene(scene_name)
        self.load_paths()

    def load_paths(self):
        self.file_names = {}
        scenes = self.data_config.scenes if self.data_config.scenes is not None else self.base_dataset.scenes
        for scene_name in scenes:
            data_dir = self.get_chunk_dir(scene_name)
            if data_dir.exists():
                self.file_names[scene_name] = [s for s in data_dir.iterdir() if s.is_file()]