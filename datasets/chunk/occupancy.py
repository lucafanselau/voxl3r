from dataclasses import dataclass, field
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
from utils.chunking import chunk_mesh, mesh_2_local_voxels

from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose

class Config(ChunkBaseDatasetConfig):
    # Grid parameters -> used to calculate center point
    grid_resolution: float = 0.02
    grid_size: list[int] = field(
        default_factory=lambda: [64, 64, 64]
    )
    folder_name_occ: str = "prepared_occ_grids"

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

    @jaxtyped(typechecker=beartype)
    def create_chunks_of_scene(
        self, base_dataset_dict: dict
    ) -> Generator[dict, None, None]:

        mesh = base_dataset_dict["mesh"]
        scene_name = base_dataset_dict["scene_name"]

        image_chunks = self.image_dataset.get_chunks_of_scene(scene_name)

        for image_chunk_path in tqdm(image_chunks, leave=True, desc=f"{scene_name}: Chunks", position=1):
            image_chunk = torch.load(image_chunk_path, weights_only=False)
            camera_dict = {
                str(Path(k).name): v
                for k, v in zip(image_chunk["images"][0], image_chunk["images"][1])
            }

            image_name = image_chunk["image_name_chunk"]
            T_cw = camera_dict[image_name]["T_cw"]
            grid_size = np.array(self.data_config.grid_size)
            size = grid_size * self.data_config.grid_resolution
            mesh_chunked, backtransformed = chunk_mesh(
                mesh.copy(), T_cw, image_chunk["center"], size, with_backtransform=True
            )

            if mesh_chunked.is_empty:
                logger.trace(
                    f"Detected empty mesh. Image name: {image_name}, Scene name: {scene_name}"
                )
                occupancy_grid = np.zeros(grid_size, dtype=np.uint8)

            else:
                _, _, T_wc = invert_pose(T_cw[:3, :3], T_cw[:3, 3])
                voxel_grid, coordinate_grid, occupancy_grid = mesh_2_local_voxels(
                    mesh_chunked,
                    center=image_chunk["center"],
                    pitch=self.data_config.grid_resolution,
                    final_dim=grid_size[0],
                    to_world_coordinates=T_wc,
                )
                occupancy_grid = occupancy_grid.astype(np.uint8)

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
            data = torch.load(file, weights_only=False)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None

        return data
