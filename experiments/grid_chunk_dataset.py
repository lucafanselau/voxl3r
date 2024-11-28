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
from experiments.surface_net_3d.projection import project_voxel_grid_to_images_seperate
from utils.chunking import (
    create_chunk,
    mesh_2_local_voxels,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose
from multiprocessing import Pool

config = load_yaml_munch("./utils/config.yaml")


@dataclass
class OccChunkDatasetConfig(SceneDatasetConfig):

    # Grid parameters
    grid_resolution: float = 0.02
    grid_size: Int[np.ndarray, "3"] = field(
        default_factory=lambda: np.array([64, 64, 64])
    )
    seq_len: int = 4

    # Chunk strategy parameters
    with_furthest_displacement: bool = False

    # Runtime config
    num_workers: int = 11
    force_prepare: bool = False


class OccChunkDataset(Dataset):
    def __init__(
        self,
        data_config: OccChunkDatasetConfig,
        transform: Optional[callable] = None,
        base_dataset: Optional[SceneDataset] = None,
    ):
        self.base_dataset = (
            base_dataset if base_dataset is not None else SceneDataset(data_config)
        )
        self.data_config = data_config
        self.transform = transform
        self.file_names = None

    def get_grid_path(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        camera = self.data_config.camera
        grid_res = self.data_config.grid_resolution
        grid_size = self.data_config.grid_size
        seq_len = self.data_config.seq_len
        with_furthest_displacement = self.data_config.with_furthest_displacement

        base_file_name = f"seq_len_{seq_len}_res_{grid_res}_size_{grid_size}"
        furthest_str = "_furthest" if with_furthest_displacement else ""

        # round float to second decimal
        chunk_size = grid_res * grid_size.astype(np.float32)
        center = "%.2f" % (chunk_size[2] * 1.15)

        path = (
            Path(self.data_config.data_dir)
            / scene_name
            / "prepared_grids"
            / camera
            / f"{base_file_name}{furthest_str}_center_{center}"
        )
        return path

    @jaxtyped(typechecker=beartype)
    def convert_scene_to_grids(
        self, base_dataset_dict: dict
    ) -> Generator[dict, None, None]:

        mesh = base_dataset_dict["mesh"]
        image_path = base_dataset_dict["path_images"]
        camera_params = base_dataset_dict["camera_params"]
        scene_name = base_dataset_dict["scene_name"]

        # important stuff from the config
        resolution = self.data_config.grid_resolution
        grid_size = self.data_config.grid_size
        seq_len = self.data_config.seq_len

        with_furthest_displacement = self.data_config.with_furthest_displacement

        image_names = list(camera_params.keys())

        chunk_size = resolution * grid_size.astype(np.float32)
        center = np.array([0.0, 0.0, chunk_size[2]])

        print("Preparing chunks for training:")
        for i in tqdm(range((len(image_names) // seq_len))):

            image_name = image_names[i * seq_len]
            data_chunk = create_chunk(
                mesh.copy(),
                image_name,
                camera_params,
                max_seq_len=seq_len,
                image_path=image_path,
                size=chunk_size,
                center=center,
                with_furthest_displacement=with_furthest_displacement,
            )

            transformation = data_chunk["camera_params"][image_name]["T_cw"]
            _, _, T_wc = invert_pose(transformation[:3, :3], transformation[:3, 3])

            if data_chunk["mesh"].is_empty:
                print(
                    f"Detected empty mesh. Skipping chunk. Image name: {image_name}, Scene name: {scene_name}"
                )
                continue

            voxel_grid, coordinate_grid, occupancy_grid = mesh_2_local_voxels(
                data_chunk["mesh"],
                center=center,
                pitch=resolution,
                final_dim=grid_size[0],
                to_world_coordinates=T_wc,
            )
            occupancy_grid = torch.from_numpy(occupancy_grid)

            # rearrange occupancy_grid to 1 W H D
            occupancy_grid = rearrange(occupancy_grid, "x y z -> 1 x y z")

            result_dict = {
                "name": scene_name,
                "resolution": resolution,
                "grid_size": grid_size,
                "chunk_size": chunk_size,
                "center": center,
                "training_data": occupancy_grid,  # (feature_grid, occupancy_grid),
                "image_name_chunk": image_name,
                "images": (
                    [str(name) for name in data_chunk["image_names"]],
                    list(data_chunk["camera_params"].values()),
                ),
            }

            yield result_dict

    def prepare_scene(self, scene_name):
        # check if preprared data exists otherwise otherwise continue
        data_dir = self.get_grid_path(scene_name)
        if data_dir.exists() and not self.data_config.force_prepare:
            return

        idx = self.base_dataset.get_index_from_scene(scene_name)

        # somehow some meshes are not available (eg. a46b21d949)
        if (
            self.base_dataset.data_dir
            / self.base_dataset.scenes[idx]
            / "scans"
            / "mesh_aligned_0.05.ply"
        ).exists() == False:
            print(f"Mesh not found for scene {scene_name}. Skipping.")
            return

        for i, scene_dict in enumerate(
            self.convert_scene_to_grids(self.base_dataset[idx])
        ):
            image_name_chunk = scene_dict["image_name_chunk"]
            data_dir = self.get_grid_path(scene_name)
            if data_dir.exists() == False:
                data_dir.mkdir(parents=True)

            torch.save(scene_dict, data_dir / f"{i}_{image_name_chunk}.pt")

    def prepare_data(self):
        scenes = (
            self.data_config.scenes
            if self.data_config.scenes is not None
            else self.base_dataset.scenes
        )

        if self.data_config.num_workers > 1:
            with Pool(self.data_config.num_workers) as p:
                p.map(self.prepare_scene, scenes)
        else:
            for scene_name in scenes:
                self.prepare_scene(scene_name)

        self.load_paths()

    def load_paths(self):
        self.file_names = {}

        for scene_name in (
            self.data_config.scenes
            if self.data_config.scenes is not None
            else self.base_dataset.scenes
        ):
            data_dir = self.get_grid_path(scene_name)
            if data_dir.exists():
                self.file_names[scene_name] = list(data_dir.iterdir())

    def get_at_idx(self, idx: int):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")
            return self.get_at_idx(idx - 1)
        if os.path.getsize(file) < 0:  # 42219083:
            print(f"File {file} is empty. Skipping.")
            return self.get_at_idx(idx - 1)

        try:
            data = torch.load(file)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1)
        occupancy_grid = data["training_data"]
        return occupancy_grid, data

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
