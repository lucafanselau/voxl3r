from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
from typing import Generator, Optional, Tuple, List, TypedDict

from beartype import beartype
from einops import rearrange
from loguru import logger
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from jaxtyping import Float, Int, Bool, jaxtyped
from torch import Tensor
import numpy as np
from tqdm import tqdm


from datasets.chunk.base import ChunkBaseDataset, ChunkBaseDatasetConfig
from datasets.chunk import image
from datasets import scene
from datasets.transforms.grid_interpolation import interpolate_grid
from utils.basic import get_default_device
from utils.chunking import chunk_mesh, compute_coordinates, mesh_2_local_voxels

from utils.data_parsing import load_yaml_munch
from utils.transformations import extract_rot_trans_batched, invert_pose, invert_pose_batched

class Config(ChunkBaseDatasetConfig):
    # Grid parameters -> used to calculate center point
    grid_resolution: float = 0.02
    grid_size: list[int] = field(
        default_factory=lambda: [64, 64, 64]
    )
    folder_name_occ: str = "prepared_occ_grids"
    force_prepare_occ: bool = False
    batch_size_occ: int = 32

class Output(TypedDict):
    scene_name: str
    image_name_chunk: str
    center: Float[np.ndarray, "3"]
    resolution: float
    grid_size: list[int]
    occupancy_grid: Float[torch.Tensor, "1 X Y Z"]

class Dataset(ChunkBaseDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: scene.Dataset,
        image_dataset: image.Dataset,
    ):
        super(Dataset, self).__init__(data_config, base_dataset)
        self.image_dataset = image_dataset
        self.data_config = data_config
        self.file_names = None
        
        # Add caching-related attributes
        self._cache = {}
        # self._cache_size = 1000  # Adjust cache size as needed
        self._cache_hits = 0
        self._cache_misses = 0

    @jaxtyped(typechecker=beartype)
    def get_chunk_dir(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        dc = self.image_dataset.data_config
        base_folder_name = f"grid_res_{self.data_config.grid_resolution}_size_{self.data_config.grid_size}_center_{dc.center_point}"

        path = (
            self.get_saving_path(scene_name)
            / self.data_config.folder_name_occ
            / base_folder_name
        )

        return path 
    
    def check_chunks_exists(self, scene_name: str) -> bool:
        chunk_dir = self.get_chunk_dir(scene_name)

        # now get all of the chunks from image dataset
        image_chunks = self.image_dataset.get_chunks_of_scene(scene_name)

        # now check for existence of all image chunks
        def check_chunk_existence(image_chunk: Path) -> bool:
            output_path = chunk_dir / image_chunk.name
            return output_path.exists() and os.path.getsize(output_path) > 0

        return chunk_dir.exists() and all(check_chunk_existence(image_chunk) for image_chunk in image_chunks)
    
    # THIS NOW EXPECTS A BATCH
    def load_prepare(self, item):
        data_dict = item
        image_names, transformations = data_dict["images"]
        transformations = torch.stack([transformation["T_cw"] for transformation in transformations])
        
        image_paths = [
            str(Path("/", *Path(img).parts[Path(img).parts.index("mnt") :]))
            for names in image_names
            for img in names
        ]

        return image_paths, transformations

    @torch.no_grad()
    def process_chunk(self, batch, batch_idx):
        
        def file_exists(idx):
                scene_name = batch["scene_name"][idx]
                saving_dir = self.get_chunk_dir(scene_name)
                if not saving_dir.exists():
                    saving_dir.mkdir(parents=True, exist_ok=True)
                file_name = saving_dir / (batch["file_name"][idx])

                return file_name.exists()

        if not self.data_config.force_prepare_mast3r and all(
            file_exists(idx) for idx in range(len(batch["scene_name"]))
        ):
            return

        voxelized_occ_grids = [self.voxelized_scene[scene_name] for scene_name in batch["scene_name"]]
        
        grid_size = self.data_config.grid_size
        center = self.data_config.center_point
        pitch = self.data_config.grid_resolution
        batch["images"][1][0]["T_cw"]
        
        coordinates = torch.from_numpy(compute_coordinates(
            np.array(grid_size),
            np.array(center),
            pitch,
            grid_size[0],
            to_world_coordinates=None,
        )).float()
        
        image_paths, transformations = self.load_prepare(batch)
        
        T_0w = transformations[0, :, :, :]
        T_w0 = invert_pose_batched(*extract_rot_trans_batched(T_0w))
        
        _3, X, Y, Z = coordinates.shape
        
        coordinates = rearrange(coordinates, "C X Y Z -> (X Y Z) C 1")
        homo_coordinates = torch.cat(
        [coordinates, torch.full((X*Y*Z, 1, 1), 1).to(coordinates)], dim=-2
        )
        
        transformed_coordinates = torch.matmul(T_w0.unsqueeze(1), homo_coordinates)[:,:,:3,:]
        
        reshaped_coordinates = rearrange(transformed_coordinates, "B (X Y Z) C 1-> B C X Y Z", X=X, Y=Y, Z=Z)
        
        occ_grids = interpolate_grid(voxelized_occ_grids, reshaped_coordinates, self.data_config.grid_resolution)
        
        for i in range(len(batch["scene_name"])):
            result_dict = {
                "scene_name": batch["scene_name"][i],
                "image_name_chunk": batch["image_name_chunk"][i],
                "center": batch["center"][i],
                "resolution": self.data_config.grid_resolution,
                "grid_size": self.data_config.grid_size,
                "occupancy_grid": occ_grids[i],
            }
            
            scene_name = batch["scene_name"][i]
            saving_dir = self.get_chunk_dir(scene_name)
            file_name = saving_dir / (batch["file_name"][i])
            if not saving_dir.exists():
                saving_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result_dict, file_name)
                            
    @torch.no_grad()
    def prepare_data(self):
        if self.data_config.skip_prepare:
            self.load_paths()
            self.on_after_prepare()
            self.prepared = True
            return

        batch_size = self.data_config.batch_size_occ

        dataloader = DataLoader(
            self.image_dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
        )
        
        self.voxelized_scene = {}
        
        for i in range(len(self.base_dataset)):
            scene_name = self.base_dataset.scenes[i]
            self.voxelized_scene[scene_name] = self.base_dataset.get_voxelized_scene(scene_name)

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            total=len(self.image_dataset) // batch_size,
        ):  
            if len(set(batch["scene_name"])) == 1:
                self.process_chunk(batch, batch_idx)
            else:
                # split into n batches where all have the same scene name
                for scene_name in set(batch["scene_name"]):
                    # get first and last index of the scene name
                    first_idx = batch["scene_name"].index(scene_name)
                    last_idx = first_idx + batch["scene_name"][first_idx:].count(scene_name)
                    scene_batch = {k: v[first_idx:last_idx] if k != "images" else ([images[first_idx:last_idx] for images in v[0]], 
                                                                                   [{ k: v[first_idx:last_idx] for k, v in images.items() } for images in v[1]])
                                                                                   for k, v in batch.items()}
                    self.process_chunk(scene_batch, batch_idx)

            
        self.voxelized_scene = None

        self.load_paths()
        self.on_after_prepare()
        self.prepared = True
        
        
    def on_after_prepare(self):
        for i in range(len(self)):
            self.get_at_idx(i)

    def get_at_idx(self, idx: int, fallback=False):
        """Get data at index with caching support"""
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        # Check cache first
        if idx in self._cache:
            self._cache_hits += 1
            return self._cache[idx]

        self._cache_misses += 1
        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]
        
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")
            return self.get_at_idx(idx - 1) if fallback else None
            
        if os.path.getsize(file) < 0:
            print(f"File {file} is empty. Skipping.")
            return self.get_at_idx(idx - 1) if fallback else None

        try:
            data = torch.load(file)
            
            # Update cache
            self._cache[idx] = data
            
            return data
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None

    def get_cache_stats(self):
        """Return cache hit/miss statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._cache),
        }

    def clear_cache(self):
        """Clear the cache and reset statistics"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

