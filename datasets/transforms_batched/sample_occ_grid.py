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
from datasets.chunk.grid_interpolation import interpolate_grid, interpolate_grid_batch
from utils.chunking import compute_coordinates
from utils.transformations import invert_pose
from torch.utils.data import default_collate

class SampleOccGridConfig(transforms.SampleCoordinateGridConfig):
    num_worker_voxelized_scenes_caching: int = 11

class SampleOccGrid(nn.Module):
    """
    Sample coordinate grid from input grid
    """

    def __init__(
        self,
        config: SampleOccGridConfig,
        *_args
    ):
        super().__init__()
        self.config = config
        self.base_dataset = scene.Dataset(config)
        self.base_dataset.prepare_data()
        
        voxelized_scenes_dict = {}
        
        print(f'Creating dict for voxelized scenes using split {config.split}')
        for scene_name in tqdm(self.base_dataset.scenes):
            if self.base_dataset.check_voxelized_scene_exists(scene_name):
                voxelized_scenes_dict[scene_name] = self.base_dataset.get_voxelized_scene(scene_name)
                
        self.encoding_sparse_indices = {k : torch.Tensor(v.encoding.sparse_indices).detach() for k, v in voxelized_scenes_dict.items()}
        self.encoding_shape = {k :torch.Tensor(v.encoding.shape).detach() for k, v in voxelized_scenes_dict.items()}
        self.voxel_grid_extents = {k : torch.Tensor(v.extents.copy()).detach() for k, v in voxelized_scenes_dict.items()}
        self.voxel_grid_transform = {k : torch.Tensor(v.transform.copy()).detach() for k, v in voxelized_scenes_dict.items()}
        self.voxel_grid_transform_inv = {k : torch.Tensor(np.linalg.inv(v.transform).copy()).detach() for k, v in voxelized_scenes_dict.items()}
            
        print('Finished dict for voxelized scenes')
  
    def forward(self, data): 
        scene_names = data["scene_name"]
        coordinate_grids = data["coordinates"].detach().cpu()
        transforms = torch.stack([self.voxel_grid_transform[scene_name] for scene_name in scene_names])
        transforms_inv = torch.stack([self.voxel_grid_transform_inv[scene_name] for scene_name in scene_names])
        sparse_encoding = [self.encoding_sparse_indices[scene_name] for scene_name in scene_names]
        B, C, X, Y, Z = coordinate_grids.shape
        
        min_voxel_location = rearrange(coordinate_grids, 'B C X Y Z -> B (X Y Z) C').min(dim=1).values
        max_voxel_location = rearrange(coordinate_grids, 'B C X Y Z -> B (X Y Z) C').max(dim=1).values
        max_size_grid = ((max_voxel_location + self.config.grid_resolution_sample) - min_voxel_location).max(dim=0).values
        
        min_voxel_location_homo = torch.cat([min_voxel_location, torch.ones(B, 1)], dim=1)
        min_voxel_idx = torch.floor(torch.bmm(transforms_inv, min_voxel_location_homo.unsqueeze(-1)))[:, :3, 0]
        min_voxel_idx[min_voxel_idx < 0] = 0
        size_voxel_grid = torch.ceil(torch.matmul(transforms_inv[:, :3, :3], max_size_grid.unsqueeze(-1)).max(dim=0).values).squeeze(-1).int()
        
        sparse_encoding = [(sparse_encoding[i] - min_voxel_idx[i]) for i in range(B)]
        sparse_encoding = [sparse_encoding[i][((0 <= sparse_encoding[i]) & (sparse_encoding[i] < size_voxel_grid)).all(dim=-1)] for i in range(B)]
        occ_voxel_grid = torch.zeros(B, size_voxel_grid[0], size_voxel_grid[1], size_voxel_grid[2])
        for i in range(B):
            occ_voxel_grid[i, sparse_encoding[i][:, 0].int(), sparse_encoding[i][:, 1].int(), sparse_encoding[i][:, 2].int()] = 1
        
        min_voxel_idx_homo = torch.cat([min_voxel_idx, torch.ones(B, 1)], dim=1).unsqueeze(-1)
        position_base_voxel = torch.matmul(transforms, min_voxel_idx_homo)[:, :3, 0].float()
        occ_voxel_drid_extent = torch.matmul(transforms[:, :3, :3], size_voxel_grid.float())
        
        occ_grids = interpolate_grid_batch(occ_voxel_grid, occ_voxel_drid_extent, position_base_voxel, coordinate_grids, self.config.scene_resolution, self.config.grid_resolution_sample)
        
        result = {
            "Y" : occ_grids.bool().detach().to(data["coordinates"]),
        }
        
        return result 
