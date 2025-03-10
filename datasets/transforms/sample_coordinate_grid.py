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

    # explicitly enable rotation, translation
    enable_rotation: bool = True
    enable_translation: bool = True


class SampleCoordinateGrid(nn.Module):
    """
    Sample coordinate grid from input grid
    """

    def __init__(self, config: SampleCoordinateGridConfig, *_args):
        super().__init__()
        self.config = config
        self.coordinate_grid_extent = np.array(config.grid_size_sample) * config.grid_resolution_sample

    # to be honest this seems more complicated to understand as expected:
    # there is an article on wikipedia tho
    # https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices
    def random_rotation_matrix(self, axis=None, max_angle=180):
        if axis is None:
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)

        max_angle_rad = np.deg2rad(max_angle)
        theta = np.random.uniform(-max_angle_rad, max_angle_rad)
        half_theta = theta / 2.0
        w = np.cos(half_theta)
        xyz = np.sin(half_theta) * axis  # This gives a vector of 3 components
        x, y, z = xyz

        # Step 4: Convert quaternion to rotation matrix
        R = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

        return R

    def random_translation(self, chunk_extent, coordinate_grid_extent):
        area_random_sample = chunk_extent - coordinate_grid_extent
        if area_random_sample.any() < 0:
            return np.array([0.0, 0.0, 0.0])
        else:
            return np.random.uniform(
                -area_random_sample/2, area_random_sample/2
            )


    def forward(self, data):
        coordinates_grid = compute_coordinates(
            np.array(self.config.grid_size_sample),
            np.array([0.0, 0.0, 0.0]),
            self.config.grid_resolution_sample,
            self.config.grid_size_sample[0],
            to_world_coordinates=None,
        )

        coordinates = rearrange(coordinates_grid, "c x y z -> (x y z) c 1")
        # TODO: no concept of "first" camera
        # T_0w = data["cameras"][0]["T_cw"]
        # _, _, T_w0 = invert_pose(T_0w[:3, :3], T_0w[:3, 3])

        # apply random rotation
        if self.training:
            if self.config.enable_rotation:
                R_random = self.random_rotation_matrix(axis=T_0w[:3, 2])
            else:
                R_random = np.eye(3)
            if self.config.enable_translation:
                t_random = self.random_translation(data["chunk_extent"], self.coordinate_grid_extent)
            else:
                t_random = np.array([0.0, 0.0, 0.0])
            T_random = np.concatenate(
                [
                    np.concatenate([R_random, np.asarray([t_random]).T], axis=1),
                    np.array([[0, 0, 0, 1]]),
                ],
                axis=0,
            )
        else:
            T_random = np.eye(4)
            t_random = np.array([0.0, 0.0, 0.0])
            T_random[:3, 3] = t_random

        coordinates = np.concatenate(
            [coordinates, np.ones((coordinates.shape[0], 1, 1))], axis=1
        )
        coordinates = T_random[:3, :] @ coordinates

        # transform coordinates to world coordinates / we assume that all pairs have the same coordinate grid
        # which is based on the extrinsics of the first camera
        coordinates = coordinates + np.expand_dims(data["chunk_center"], -1)
        coordinates_grid = rearrange(
            coordinates,
            "(x y z) c 1 -> c x y z",
            x=self.config.grid_size_sample[0],
            y=self.config.grid_size_sample[0],
            z=self.config.grid_size_sample[0],
        )

        data["coordinates"] = torch.from_numpy(coordinates_grid).float()

        if not "verbose" in data.keys():
            data["verbose"] = {}

        data["verbose"]["grid_size"] = self.config.grid_size_sample
        data["verbose"]["resolution"] = self.config.grid_resolution_sample
        data["verbose"]["center"] = data["chunk_center"] + t_random.flatten()
        data["verbose"]["T_random"] = T_random

        return data
