from multiprocessing import Pool
import os
from pathlib import Path
from typing import Optional, Union, List, Any

import io
from PIL import Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from datasets import scene
from loguru import logger

from datasets.chunk.mast3r import update_camera_intrinsics
from datasets.chunk.pair_matching import PairMatching

from .image import Dataset as ImageDataset, Config as ImageDatasetConfig, Output as ImageOutput

from extern.mast3r.dust3r.dust3r.utils.image import load_images

class Config(ImageDatasetConfig):
    pair_matching: str = "first_centered"
    compressed_img_folder: str = "compressed_images"
    load_last_cached: bool = True

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
        split: str = "train",
    ):
        super().__init__(data_config, base_dataset)
        self.data_config = data_config
        self.pair_matching = PairMatching[data_config.pair_matching]()
        self.cache = {}
        self.split = split
        
    def compress_tensor_to_png(self, tensor):
        """
        Convert a uint8 tensor (shape CxHxW or HxWxC) to a PNG-encoded bytes object.
        """
        # Convert tensor to NumPy array if needed
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()
        # If tensor shape is C x H x W, convert to H x W x C
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            tensor = np.transpose(tensor, (1, 2, 0))
        # Ensure the array is uint8
        tensor = tensor.astype(np.uint8)
        # Create a PIL Image
        image = Image.fromarray(tensor)
        # Save the image to a bytes buffer in PNG format (lossless)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()

    def decompress_png_to_tensor(self, png_bytes):
        """
        Convert PNG-encoded bytes back to a uint8 tensor.
        """
        buffer = io.BytesIO(png_bytes)
        image = Image.open(buffer)
        np_image = np.array(image)
        # If needed, convert from H x W x C to C x H x W
        if np_image.ndim == 3 and np_image.shape[2] == 3:
            np_image = np.transpose(np_image, (2, 0, 1))
        return torch.from_numpy(np_image).float()


    # disable loading into cache
    def on_after_prepare(self): 
        
        print(f'Loading images into cache for {self.split}-dataset')
        
        if(Path(self.data_config.data_dir) / self.data_config.storage_preprocessing / f'last_cache_{self.split}.pt').exists() and self.data_config.load_last_cached:
            self.cache = torch.load(Path(self.data_config.data_dir) / self.data_config.storage_preprocessing / f'last_cache_{self.split}.pt')
        else:
            for i in tqdm(range(len(self))):
                self.get_at_idx(i)
            torch.save(self.cache, Path(self.data_config.data_dir) / self.data_config.storage_preprocessing / f'last_cache_{self.split}.pt')
                        
        print(f'Finished loading images into cache for {self.split}-dataset')

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
            chunk_dict = torch.load(file_path)
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
        
        mast3r_size = (512, 336)
        images_tensor = self._load_images_into_tensor(image_paths, size=mast3r_size[0])

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

    def _load_images_into_tensor(self, image_paths: List[Union[str, Path]], size=512) -> torch.Tensor:
        """
        Helper method: given a list of image paths, load them into a single torch tensor.
        The final shape will be [N, C, H, W].
        """
        loaded_images = []
        for img_path in image_paths:
            
            if not os.path.isfile(img_path):
                logger.warning(f"Image path {img_path} not found, skipping.")
                continue
            
            if img_path in self.cache:
                loaded_images.append(self.decompress_png_to_tensor(self.cache[img_path]))
                continue
            
            saving_path = Path(img_path).parents[1] / self.data_config.compressed_img_folder / Path(img_path).name
            
            if saving_path.exists():
                self.cache[img_path] = torch.load(saving_path)
                loaded_images.append(self.decompress_png_to_tensor(self.cache[img_path]))
                continue
            
            img = load_images([img_path], size, verbose=False)[0]["img"][0]
            img = ((img + 1) / 2) * 255
            compressed_img = self.compress_tensor_to_png(img)
            self.cache[img_path] = compressed_img
            
            if not saving_path.parents[0].exists():
                os.mkdir(saving_path.parents[0])
                
            torch.save(compressed_img, Path(img_path).parents[1] / self.data_config.compressed_img_folder / Path(img_path).name)
            loaded_images.append(img.float())
            

        if len(loaded_images) == 0:
            # If no images found, return an empty 4D tensor: [0, 3, 0, 0]
            return torch.empty((0, 3, 0, 0))

        images_tensor = torch.stack(loaded_images, dim=0)
        return images_tensor
