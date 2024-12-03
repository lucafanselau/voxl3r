from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Generator, Optional, Tuple, List

import PIL
from PIL.ImageOps import exif_transpose
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
from experiments.mast3r import load_model, predict
from experiments.mast3r_baseline.module import Mast3rBaselineConfig
from experiments.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
from experiments.surface_net_3d.projection import project_voxel_grid_to_images_seperate
from extern.mast3r.dust3r.dust3r.utils.image import _resize_pil_image
from utils.chunking import (
    create_chunk,
    mesh_2_local_voxels,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose
from multiprocessing import Pool

config = load_yaml_munch("./utils/config.yaml")



class Mast3rChunkDataset(OccChunkDataset):
    def __init__(
        self,
        data_config: OccChunkDatasetConfig,
        transform: Optional[callable] = None,
        base_dataset: Optional[SceneDataset] = None,
        mast3r_results_dir: Optional[str] = "mast3r_preprocessed"
    ):
        super(Mast3rChunkDataset, self).__init__(data_config, transform=None, base_dataset=base_dataset)
        self.data_config = data_config
        self.transform = transform
        self.mast3r_results_dir = mast3r_results_dir


    def prepare_data(self):
        
        super(Mast3rChunkDataset, self).prepare_data()
        
        model = load_model(Mast3rBaselineConfig().model_name)
        
        for idx in range(super(Mast3rChunkDataset, self).__len__()):
            item = super(Mast3rChunkDataset, self).get_at_idx(idx) # no need for transform
            occ, data_dict = item
            scene_name = data_dict["name"]
            image_paths = [str(Path("/", *Path(img).parts[Path(img).parts.index("mnt"):])) for img in data_dict["images"][0]]
            image_dicts = data_dict["images"][1]
            size = 512
            
            pairwise_predictions = predict(model, None, image_paths=image_paths, image_size=size)
            
            # need to adapt K to new image format
            img = exif_transpose(PIL.Image.open(image_paths[0])).convert('RGB')
            
            # account for scaling
            W_old, H_old = img.size
            
            if size == 224:
                long_edge_size = round(size * max(W_old/H_old, H_old/W_old)) # resize short side to 224 (then crop)
            else:
                long_edge_size = size # resize long side to 512
                
            W_new, H_new = _resize_pil_image(img, long_edge_size).size
            s_x = W_new / W_old
            s_y = H_new / H_old
            
            # account for cropping
            cx, cy = W_new//2, H_new//2
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            delta_x_left, delta_y_top, _, _ = (cx-halfw, cy-halfh, cx+halfw, cy+halfh)
        
            K_old = image_dicts[0]["K"]
            K_new = np.array(
                [[K_old[0][0] * s_x, 0.0, K_old[0][2] * s_x - delta_x_left],
                [0.0, K_old[1][1] * s_y, K_old[1][2] * s_y - delta_y_top],
                [0.0, 0.0, 1.0]])
            
            for image_dict in image_dicts:
                image_dict["K"] = K_new
 
            master_chunk_dict = {
                "name": scene_name,
                "resolution": data_dict["resolution"],
                "grid_size":  data_dict["grid_size"],
                "chunk_size": data_dict["chunk_size"],
                "center": data_dict["center"],
                "training_data": occ,
                "image_name_chunk": data_dict["image_name_chunk"],
                "images": (
                    [image_paths],
                    image_dict,
                ),
                "pairwise_predictions": pairwise_predictions,
                
            }
            
            data_dir = super(Mast3rChunkDataset, self).get_grid_path(scene_name) / self.mast3r_results_dir 
            if data_dir.exists() == False:
                data_dir.mkdir(parents=True)

            torch.save(master_chunk_dict, data_dir / (f"{idx}_" + str(data_dict["image_name_chunk"]) + ".pt"))
  
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
