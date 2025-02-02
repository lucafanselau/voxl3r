from typing import Optional
from jaxtyping import jaxtyped, Bool, Float
import numpy as np
from torch import Tensor
import torch

from utils.transformations import extract_rot_trans, invert_pose
from . import base

import pyvista as pv

class Config(base.Config):
    pass


class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)
        
    def add_colored_pointcloud_from_data_dict(self, smeared_images_dict: dict) -> None:
        """

        Args:
            image_dict (dict): smeared image dict of image that is to be visualized
        """
        
        coordinates = smeared_images_dict["verbose"]["data_dict"]["coordinates"].reshape(3, -1).T
        occ_mask = smeared_images_dict["Y"].flatten()
        #occ_mask = torch.ones_like(occ_mask, dtype=torch.bool)
        points_of_interests = coordinates[occ_mask]
        
        if "T_random" in smeared_images_dict["verbose"]["data_dict"].keys():
            print("T_random was: \n" + str(smeared_images_dict["verbose"]["data_dict"]["T_random"]))
        
        
        for colors in smeared_images_dict["X"]:
            for i, pair in enumerate(colors):
                
                color = pair.reshape(3, -1).T[occ_mask]
                color[color < 0] = 0
                color = (color * 255).int()
                
                point_cloud = pv.PolyData(points_of_interests.numpy())
                # Assign the RGB colors to a new point array named 'RGB'
                point_cloud["RGB"] = color.numpy()

                # Add the mesh with per-point colors; note the use of the 'scalars' parameter and rgb=True.
                self.plotter.add_mesh(
                    point_cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=10,
                    render_points_as_spheres=True  # optional, for improved visual quality
                )
                break
                
            break

        