from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, List, TypedDict

from loguru import logger
import torch
from jaxtyping import Float
import numpy as np
from tqdm import tqdm

from datasets.scene import (
    Dataset,
    Config,
)
from .base import ChunkBaseDataset, ChunkBaseDatasetConfig
from utils.chunking import (
    retrieve_images_for_chunk,
)
from utils.data_parsing import load_yaml_munch
from multiprocessing import Pool


class Config(ChunkBaseDatasetConfig):
    # Image Config
    seq_len: int = 4
    with_furthest_displacement: bool = False
    center_point: Float[np.ndarray, "3"] = field(
        default_factory=lambda: np.array(
            [0.0, 0.0, 1.28]
        )  # center point of chunk in camera coodinates
    )

    folder_name_image: str = "corresponding_images"

class Output(TypedDict):
    scene_name: str
    images: Tuple[List[str], List[Dict[str, float]]]
    center: Float[np.ndarray, "3"]
    image_name_chunk: str

class Dataset(ChunkBaseDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: Dataset,
        transform: Optional[callable] = None,
    ):
        super(Dataset, self).__init__(data_config, base_dataset)
        self.data_config = data_config
        self.transform = transform
        self.file_names = None
        self.image_cache = {}

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

    def get_chunks_of_scene(self, scene_name: str) -> List[Path]:
        chunk_dir = self.get_chunk_dir(scene_name)
        files = [s for s in chunk_dir.iterdir() if s.is_file()]
        return files
    
    def check_chunks_exists(self, scene_name: str) -> bool:
        chunk_dir = self.get_chunk_dir(scene_name)
        # here existance is enough
        return chunk_dir.exists()

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

        for i in tqdm(range((len(image_names) // seq_len)), leave=False):
            try:
                image_name = image_names[i * seq_len]
                camera_params_chunk, image_names_chunk = retrieve_images_for_chunk(
                    camera_params,
                    image_name,
                    seq_len,
                    center,
                    with_furthest_displacement,
                    image_path,
                )

                result_dict = {
                    "scene_name": scene_name,
                    "center": center,
                    "image_name_chunk": image_name,
                    "images": (
                        [str(name) for name in image_names_chunk],
                        list(camera_params_chunk.values()),
                    ),
                }

                yield result_dict, image_name
            except Exception as e:
                logger.error(f"[ChunkImageDataset] Error creating chunk for scene {scene_name}: {e}")
                continue
            
    def on_after_prepare(self):
        for i in range(len(self)):
            self.get_at_idx(i)

    def get_at_idx(self, idx: int, fallback: Optional[bool] = False):
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
            
            if str(file) not in self.image_cache:
                self.image_cache[str(file)] = torch.load(file)
                
            data_dict = self.image_cache[str(file)]
        except Exception as e:
            print(f"Error loading file {file}: {e}")
            return self.get_at_idx(idx - 1) if fallback else None

        return data_dict

    