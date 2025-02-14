import os
from pathlib import Path
from typing import Optional, Union, List, Any

import torch
import torchvision
from datasets import scene
from loguru import logger

from datasets.chunk.pair_matching import PairMatching

from .image import Dataset as ImageDataset, Config as ImageDatasetConfig, Output as ImageOutput

class Config(ImageDatasetConfig):
    pair_matching: str = "first_centered"

class Output(ImageOutput):
    images_tensor: torch.Tensor
    pairs_image_names: list[tuple[str, str]]


class Dataset(ImageDataset):
    """
    This dataset simply loads images and metadata from the .pt files produced by the chunk.image.Dataset.
    It does not generate or prepare chunks, but rather just reads them and converts images to torch tensors.
    """

    def __init__(
        self,
        data_config: Config,
        base_dataset: scene.Dataset,
    ):
        super().__init__(data_config, base_dataset)
        self.pair_matching = PairMatching[data_config.pair_matching]()


    # disable loading into cache
    def on_after_prepare(self):
        pass

    def get_at_idx(self, idx: int, fallback: bool = False) -> dict:
        """
        Reads the corresponding .pt file, loads and accumulates images into a torch tensor,
        and returns a dictionary with 'images_tensor', 'scene_name', etc.
        """
        if self.file_names is None:
            raise ValueError("No file names loaded. Did you forget to call prepare_data()?")

        # Flatten all scene file lists into one index
        all_files = [file for files in self.file_names.values() for file in files]
        if idx < 0 or idx >= len(all_files):
            raise IndexError(f"Index {idx} out of range for dataset length {len(all_files)}.")

        file_path = all_files[idx]
        if not file_path.is_file():
            logger.warning(f"File {file_path} does not exist or is not a file.")
            return {}

        # Load existing chunk dictionary from disk
        try:
            chunk_dict = torch.load(file_path, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load chunk file {file_path}: {e}")
            if fallback and idx > 0:
                logger.info(f"Falling back to idx {idx-1}.")
                return self.get_at_idx(idx - 1, fallback=False)
            else:
                return {}

        # chunk_dict is something like:
        # {
        #   "scene_name": scene_name,
        #   "images": [list_of_image_paths],
        #   "cameras": [list_of_camera_params],
        #   "image_name_chunk": ...
        #   ...
        # }

        image_paths = chunk_dict.get("images", [])
        images_tensor = self._load_images_into_tensor(image_paths)

        seq_len = len(image_paths)
        pair_indices = self.pair_matching(seq_len)

        image_names = [str(Path(name).name) for name in image_paths]

        pairs_image_names = [(image_names[pair_idx[0]], image_names[pair_idx[1]]) for pair_idx in pair_indices]

        pairwise_prediction = images_tensor[pair_indices]

        out_dict = {
            **chunk_dict,
            "pairs_indices": pair_indices,
            "pairs_image_names": pairs_image_names,
            "pairwise_prediction": pairwise_prediction,
        }

        return out_dict

    def _load_images_into_tensor(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """
        Helper method: given a list of image paths, load them into a single torch tensor.
        The final shape will be [N, C, H, W].
        """
        loaded_images = []
        for img_path in image_paths:
            if not os.path.isfile(img_path):
                logger.warning(f"Image path {img_path} not found, skipping.")
                continue

            # Load image with torchvision for convenience
            img = torchvision.io.read_image(str(img_path)).float()
            loaded_images.append(img)

        if len(loaded_images) == 0:
            # If no images found, return an empty 4D tensor: [0, 3, 0, 0]
            return torch.empty((0, 3, 0, 0))

        images_tensor = torch.stack(loaded_images, dim=0)
        return images_tensor
