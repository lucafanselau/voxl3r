from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Generator, Optional, Tuple, List

from beartype import beartype
import torch
import lightning.pytorch as pl
from jaxtyping import Float, Int, Bool, jaxtyped
from torch import Tensor
import numpy as np
from tqdm import tqdm

from dataset import (
    SceneDataset,
    SceneDatasetConfig,
)
from experiments.chunk_dataset import ChunkDataset
from utils.chunking import (
    retrieve_images_for_chunk,
)
from utils.data_parsing import load_yaml_munch
from multiprocessing import Pool

config = load_yaml_munch("./utils/config.yaml")


@dataclass
class ImageChunkDatasetConfig(SceneDatasetConfig):

    # Image Config
    seq_len: int = 4
    with_furthest_displacement: bool = False
    center_point: Float[np.ndarray, "3"] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.28]) # center point of chunk in camera coodinates
    )
    
    folder_name_image: str = "corresponding_images"

    # Runtime Config
    num_workers: int = 1
    force_prepare: bool = False


class ImageChunkDataset(ChunkDataset):
    def __init__(
        self,
        data_config: ImageChunkDatasetConfig,
        transform: Optional[callable] = None,
        base_dataset: Optional[SceneDataset] = None,
    ):
        super(ImageChunkDataset, self).__init__(data_config, base_dataset)
        self.base_dataset = (
            base_dataset if base_dataset is not None else SceneDataset(data_config)
        )
        self.data_config = data_config
        self.transform = transform
        self.file_names = None

    @jaxtyped(typechecker=beartype)
    def get_chunk_dir(self, scene_name: str) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        dc = self.data_config
        base_folder_name = f"seq_len_{dc.seq_len}_furthest_{dc.with_furthest_displacement}_center_{dc.center_point}"

        path = (
            self.get_saving_path(scene_name)
            / self.data_config.folder_name_image
            / base_folder_name
        )
        
        return path


    @jaxtyped(typechecker=beartype)
    def get_chunks_of_scene(
        self, scene_name: str
    ) -> List[Path]:
        chunk_dir = self.get_chunk_dir(scene_name)
        files = [s for s in chunk_dir.iterdir() if s.is_file()]
        return sorted(files, key=lambda f: int(f.name.split("_")[0]))
        
    @jaxtyped(typechecker=beartype)
    def create_chunks_of_scene(
        self, base_dataset_dict: dict
    ) -> Generator[dict, None, None]:

        image_path = base_dataset_dict["path_images"]
        camera_params = base_dataset_dict["camera_params"]
        scene_name = base_dataset_dict["scene_name"]

        seq_len = self.data_config.seq_len
        with_furthest_displacement = self.data_config.with_furthest_displacement
        image_names = list(camera_params.keys())
        center = self.data_config.center_point

        print("Preparing image chunks for training:")
        for i in tqdm(range((len(image_names) // seq_len))):

            image_name = image_names[i * seq_len]
            camera_params_chunk, image_names_chunk = retrieve_images_for_chunk(camera_params, image_name, seq_len, center, with_furthest_displacement, image_path)
          
            result_dict = {
                "scene_name": scene_name,
                "center": center,
                "image_name_chunk": image_name,
                "images": (
                    [str(name) for name in image_names_chunk],
                    list(camera_params_chunk.values()),
                ),
            }

            yield result_dict, None

    @jaxtyped(typechecker=beartype)
    def get_at_idx(self, idx: int, fallback: Optional[bool]=False):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")

            return self.get_at_idx(idx - 1) if fallback else None
        if os.path.getsize(file) < 0: 
            print(f"File {file} is empty. Skipping.")
            return self.get_at_idx(idx - 1) if fallback else None

        try:
            data_dict = torch.load(file)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None
        
        return data_dict

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
