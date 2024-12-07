from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Generator, Optional, Tuple, List

from beartype import beartype
from einops import rearrange
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split, Dataset
from jaxtyping import Float, Int, Bool, jaxtyped
from torch import Tensor
import numpy as np
from tqdm import tqdm


from dataset import (
    SceneDataset,
    SceneDatasetConfig,
)
from experiments.chunk_dataset import ChunkDataset
from experiments.image_chunk_dataset import ImageChunkDataset, ImageChunkDatasetConfig
from experiments.surface_net_3d.projection import project_voxel_grid_to_images_seperate
from utils.chunking import (
    chunk_mesh,
    create_chunk,
    mesh_2_local_voxels,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose
from multiprocessing import Pool

config = load_yaml_munch("./utils/config.yaml")


@dataclass
class OccChunkDatasetConfig(ImageChunkDatasetConfig):
    # Grid parameters -> used to calculate center point
    grid_resolution: float = 0.02
    grid_size: Int[np.ndarray, "3"] = field(
        default_factory=lambda: np.array([64, 64, 64])
    )
    folder_name_occ: str = "prepared_occ_grids"
    

class OccChunkDataset(ChunkDataset):
    def __init__(
        self,
        data_config: OccChunkDatasetConfig,
        transform: Optional[callable] = None,
        base_dataset: Optional[SceneDataset] = None,
    ):
        
        self.base_dataset = (
            base_dataset if base_dataset is not None else SceneDataset(data_config)
        )
        self.image_dataset = ImageChunkDataset(data_config, base_dataset=self.base_dataset)
        self.data_config = data_config
        self.transform = transform
        self.file_names = None

    @jaxtyped(typechecker=beartype)
    def get_chunk_dir(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        dc = self.data_config
        base_folder_name = f"grid_res_{dc.grid_resolution}_size_{dc.grid_size}_center_{dc.center_point}"

        path = (
            self.get_saving_path(scene_name)
            / self.data_config.folder_name_occ
            / base_folder_name
        )
        
        return path
    
    def create_chunk_from_path(image_chunk_path: str):
        image_chunk = torch.load(image_chunk_path)
        camera_dict = {str(Path(k).name): v for k, v in zip(image_chunk["images"][0], image_chunk["images"][1])}
        
        image_name = image_chunk["image_name_chunk"]
        T_cw = camera_dict[image_name]["T_cw"]
        size = self.data_config.grid_size*self.data_config.grid_resolution
        mesh_chunked, backtransformed = chunk_mesh(x
            mesh.copy(), T_cw, image_chunk["center"], size, with_backtransform=True
        )
        
        if mesh_chunked.is_empty:
            print(
                f"Detected empty mesh. Image name: {image_name}, Scene name: {scene_name}"
            )
            occupancy_grid = np.zeros(self.data_config.grid_size)
        
        else:
            _, _, T_wc = invert_pose(T_cw[:3, :3], T_cw[:3, 3])
            voxel_grid, coordinate_grid, occupancy_grid = mesh_2_local_voxels(
                mesh_chunked,
                center=image_chunk["center"],
                pitch=self.data_config.grid_resolution,
                final_dim=self.data_config.grid_size[0],
                to_world_coordinates=T_wc,
            )
            
        occupancy_grid = torch.from_numpy(occupancy_grid)
        occupancy_grid = rearrange(occupancy_grid, "x y z -> 1 x y z")
        
        result_dict = {
            "scene_name": scene_name,
            "image_name_chunk": image_name,
            "center": image_chunk["center"],
            "resolution": self.data_config.grid_resolution,
            "grid_size": self.data_config.grid_size,
            "occupancy_grid": occupancy_grid,
        }
        
        
    @jaxtyped(typechecker=beartype)
    def create_chunks_of_scene(
        self, base_dataset_dict: dict
    ) -> Generator[dict, None, None]:    

        mesh = base_dataset_dict["mesh"]
        scene_name = base_dataset_dict["scene_name"]
        
        image_chunks = self.image_dataset.get_chunks_of_scene(scene_name)
        
        print("Preparing occupancy chunks for training:")
        for image_chunk_path in tqdm(image_chunks):
            image_chunk = torch.load(image_chunk_path)
            camera_dict = {str(Path(k).name): v for k, v in zip(image_chunk["images"][0], image_chunk["images"][1])}
            
            image_name = image_chunk["image_name_chunk"]
            T_cw = camera_dict[image_name]["T_cw"]
            size = self.data_config.grid_size*self.data_config.grid_resolution
            mesh_chunked, backtransformed = chunk_mesh(
                mesh.copy(), T_cw, image_chunk["center"], size, with_backtransform=True
            )
            
            if mesh_chunked.is_empty:
                print(
                    f"Detected empty mesh. Image name: {image_name}, Scene name: {scene_name}"
                )
                occupancy_grid = np.zeros(self.data_config.grid_size)
            
            else:
                _, _, T_wc = invert_pose(T_cw[:3, :3], T_cw[:3, 3])
                voxel_grid, coordinate_grid, occupancy_grid = mesh_2_local_voxels(
                    mesh_chunked,
                    center=image_chunk["center"],
                    pitch=self.data_config.grid_resolution,
                    final_dim=self.data_config.grid_size[0],
                    to_world_coordinates=T_wc,
                )
                
            occupancy_grid = torch.from_numpy(occupancy_grid)
            occupancy_grid = rearrange(occupancy_grid, "x y z -> 1 x y z")
            
            result_dict = {
                "scene_name": scene_name,
                "image_name_chunk": image_name,
                "center": image_chunk["center"],
                "resolution": self.data_config.grid_resolution,
                "grid_size": self.data_config.grid_size,
                "occupancy_grid": occupancy_grid,
            }        

            yield result_dict, image_chunk["file_name"]

    def get_at_idx(self, idx: int, fallback=False):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")

            return self.get_at_idx(idx - 1) if fallback else None
        if os.path.getsize(file) < 0:  # 42219083:
            print(f"File {file} is empty. Skipping.")
            return self.get_at_idx(idx - 1) if fallback else None

        try:
            data = torch.load(file)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None
        
        return data

    def __getitem__(self, idx):
        result = self.get_at_idx(idx)

        if self.transform is not None:
            result = self.transform(result, idx)

        return result

    def __len__(self):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )
        return sum([len(self.file_names[scene_name]) for scene_name in self.file_names])
