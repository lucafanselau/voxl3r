from dataclasses import field
from einops import rearrange
import numpy as np
from sympy import Float
import torch
import torch.nn as nn
from datasets.chunk import image
from utils.chunking import compute_coordinates
from utils.transformations import invert_pose

class SampleCoordinateGridConfig(image.Config):
    grid_resolution_sample: float
    grid_size_sample: list[int]
    

class SampleCoordinateGrid(nn.Module):
    """
    Sample coordinate grid from input grid
    """

    def __init__(
        self,
        config: SampleCoordinateGridConfig,
        *_args
    ):
        super().__init__()
        self.config = config
        chunk_cube_size_image = config.grid_resolution * np.array(config.grid_size)
        chunk_cube_size_sampled = config.grid_resolution_sample * np.array(config.grid_size_sample)
        norm_center_corner_sampled = np.linalg.norm(chunk_cube_size_sampled / 2.0)
        
        # how much can I move the smaller chunk in space such that all
        # corners are still in the image chunk (in th worst case so also considering the rotation)
        self.translation_margin =  (chunk_cube_size_image/2 - norm_center_corner_sampled)
        self.translation_margin[self.translation_margin < 0] = 0
        self.initial_center_point = np.array(self.config.center_point)
        
    # to be honest this seems more complicated to understand as expected:
    # there is an article on wikipedia tho
    # https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices
    def random_rotation_matrix(self, max_angle=180):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        
        max_angle_rad = np.deg2rad(max_angle)
        theta = np.random.uniform(-max_angle_rad, max_angle_rad)
        half_theta = theta / 2.0
        w = np.cos(half_theta)
        xyz = np.sin(half_theta) * axis  # This gives a vector of 3 components
        x, y, z = xyz
        
        # Step 4: Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    # def random_rotation_matrix(self, max_angle=15):
    #     axis = np.random.randn(3)
    #     axis /= np.linalg.norm(axis)
        
    #     max_angle_rad = np.deg2rad(max_angle)
    #     theta = np.random.uniform(-max_angle_rad, max_angle_rad)
    #     half_theta = theta / 2.0
    #     w = np.cos(half_theta)
    #     xyz = np.sin(half_theta) * axis  # This gives a vector of 3 components
    #     x, y, z = xyz
        
    #     # Step 4: Convert quaternion to rotation matrix
    #     R = np.array([
    #         [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
    #         [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
    #         [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    #     ])
        
    #     return R
    
    def random_translation(self):
        t_delta = self.translation_margin * np.random.uniform(-np.array([1,1,1]), np.array([1, 1, 1]))
        return (self.initial_center_point + t_delta).reshape(3, 1)
  
    def forward(self, data):
        coordinates_grid = compute_coordinates(
            np.array(self.config.grid_size_sample),
            np.array([0.0, 0.0, 0.0]),
            self.config.grid_resolution_sample,
            self.config.grid_size_sample[0],
            to_world_coordinates=None,
        )
        
        coordinates = rearrange(coordinates_grid, "c x y z -> (x y z) c 1")
        
        # apply random rotation
        
        if self.config.split == "train":
            R_random = self.random_rotation_matrix()
            t_random = self.random_translation()
            T_random = np.concatenate([np.concatenate([R_random, t_random], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
        else:
            T_random = np.eye(4)
            t_random = self.initial_center_point
            T_random[:3, 3] = t_random
            
        coordinates = np.concatenate([coordinates, np.ones((coordinates.shape[0], 1, 1))], axis=1)
        coordinates = T_random @ coordinates
        
        # transform coordinates to world coordinates / we assume that all pairs have the same coordinate grid
        # which is based on the extrinsics of the first camera
        T_0w = data["cameras"][0]["T_cw"]
        _, _, T_w0 = invert_pose(T_0w[:3, :3], T_0w[:3, 3])
        
        coordinates = T_w0[:3, :] @ coordinates
        coordinates_grid = rearrange(
            coordinates,
            "(x y z) c 1 -> c x y z",
            x=self.config.grid_size_sample[0],
            y=self.config.grid_size_sample[0],
            z=self.config.grid_size_sample[0],
        )
        
        data["coordinates"] = torch.from_numpy(coordinates_grid).float()
        
        # used for debugging purposes
        data["grid_size"] = self.config.grid_size_sample
        data["resolution"] = self.config.grid_resolution_sample
        data["center"] = t_random.flatten()
        data["T_random"] = T_random
        
        return data