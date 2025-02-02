from dataclasses import field
from multiprocessing import Manager, Pool
from multiprocessing import managers
from einops import rearrange
import numpy as np
from sympy import Float
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import scene, transforms
from datasets.chunk import image
from datasets.chunk.grid_interpolation import interpolate_grid
from utils.chunking import compute_coordinates
from utils.transformations import invert_pose

class SampleOccGridConfig(transforms.SampleCoordinateGridConfig):
    num_worker_voxelized_scenes_caching: int = 11

class SampleOccGrid(nn.Module):
    """
    Sample coordinate grid from input grid
    """

    def __init__(
        self,
        config: SampleOccGridConfig,
        shared_dict: managers.DictProxy
    ):
        super().__init__()
        self.config = config
        self.base_dataset = scene.Dataset(config)
        self.base_dataset.prepare_data()
        
        self.voxelized_scenes_shared_dict = shared_dict
        print('Creating shared_dict')
        with Pool(self.config.num_worker_voxelized_scenes_caching) as p:
            with tqdm(total=self.base_dataset.__len__(), position=0, leave=True) as pbar:
                for _ in p.imap_unordered(self.add_scene_to_shared_dict, self.base_dataset.scenes):
                    pbar.update()
            
        print('Finished creating shared_dict')
    
    def add_scene_to_shared_dict(self, scene_name):
        if scene_name not in self.voxelized_scenes_shared_dict:
            if self.base_dataset.check_voxelized_scene_exists(scene_name):
                self.voxelized_scenes_shared_dict[scene_name] = self.base_dataset.get_voxelized_scene(scene_name)
  
    def forward(self, data): 
        scene_name = data["scene_name"]
        
        if scene_name not in self.voxelized_scenes_shared_dict:
            print('Adding {} to shared_dict'.format(scene_name))
            self.voxelized_scenes_shared_dict[scene_name] = self.base_dataset.get_voxelized_scene(scene_name)

        occ_grid_sampled = interpolate_grid(
            [self.voxelized_scenes_shared_dict[scene_name]],
            data["coordinates"].unsqueeze(0),
            self.config.grid_resolution_sample
        )
        data["occupancy_grid"] = occ_grid_sampled.squeeze(0)
        return data