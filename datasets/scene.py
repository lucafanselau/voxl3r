from multiprocessing import Pool
from pathlib import Path
import time
from typing import Optional
from typing import List
from loguru import logger
from matplotlib import spines
import pandas as pd
import tqdm
from utils.config import BaseConfig
import torch
from torch.utils.data import Dataset
import numpy as np

import pyvista as pv
import trimesh
from torch import nn

from common.scene_release import ScannetppScene_Release
from iphone.prepare_iphone_data import (
    extract_depth,
    extract_masks,
    extract_rgb,
)
from utils.data_parsing import (
    get_camera_params,
)

class Config(BaseConfig):
    data_dir: str = "datasets/scannetpp/data"
    camera: str = "dslr"
    scenes: Optional[List[str]] = None
    
    # voxelization parameters
    storage_preprocessing_voxelized_scenes: str = "preprocessed_voxel_grids"
    num_workers_voxelization: int = 11
    force_prepare_voxelize: bool = False
    scene_resolution: float = 0.01
    return_voxelized: bool = True


class Dataset(Dataset):
    def __init__(
        self,
        data_config: Config,
    ):
        self.data_config = data_config
        self.camera = data_config.camera
        self.data_dir = Path(data_config.data_dir)
        self.scenes = (
            [x.name for x in self.data_dir.glob("*") if x.is_dir()]
            if data_config.scenes is None
            else data_config.scenes
        )
        
    def get_saving_path(self, scene_name: str) -> Path:
        return (
            Path(self.data_config.data_dir)
            / self.data_config.storage_preprocessing_voxelized_scenes
            / scene_name
            / self.data_config.camera
        )
        
    def get_voxelized_scene_dir(self, scene_name: str, resolution: float) -> Path:
        """Creates the path for storing grid files based on configuration parameters"""
        base_folder_name = f"grid_res_{resolution}"
        path = (
            Path(self.data_config.data_dir)
            / self.data_config.storage_preprocessing_voxelized_scenes
            / scene_name
            / base_folder_name
        )
        return path 
    
    def load_paths(self):
        self.file_names = {}

        for scene_name in self.scenes:
            data_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)
            if data_dir.exists():
                self.file_names[scene_name] = [
                    s for s in data_dir.iterdir() if s.is_file()
                ]

        
    def prepare_data(self):
        if self.data_config.num_workers_voxelization > 1:
            with Pool(self.data_config.num_workers_voxelization) as p:
                with tqdm.tqdm(total=len(self.scenes), position=0, leave=True) as pbar:
                    for _ in p.imap_unordered(self.prepare_scene, self.scenes):
                        pbar.update()
        else:
            for scene_name in tqdm.tqdm(self.scenes, leave=False):
                self.prepare_scene(scene_name)
        self.load_paths()
        self.prepared = True
        
    def check_voxelized_scene_exists(self, scene_name: str) -> bool:
        voxelized_scene_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)
        return voxelized_scene_dir.exists()

    def get_file_name_voxelized_scene(self, scene_name: str):
        return f"{scene_name}.pt"
        
    def prepare_scene(self, scene_name: str):

        data_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)

        if self.check_voxelized_scene_exists(scene_name) and not self.data_config.force_prepare_voxelize:
            # we need to check if all of the chunks for this scene are present
            logger.trace(f"Chunks for scene {scene_name} already exist. Skipping.")
            return
        
        mesh_path = (
            self.data_dir
            / scene_name
            / "scans"
            / "mesh_aligned_0.05.ply"
        )
        if not mesh_path.exists():
            print(f"Mesh not found for scene {scene_name}. Skipping.")
            return

        voxelized_scene = self.voxelize_scene(scene_name, resolution=self.data_config.scene_resolution)
        
        if not data_dir.exists():
            data_dir.mkdir(parents=True)


        torch.save(voxelized_scene, data_dir / self.get_file_name_voxelized_scene(scene_name))
        
    def voxelize_scene(self, scene_name, resolution=0.01):
        idx = self.get_index_from_scene(scene_name)
        voxel_grid = trimesh.voxel.creation.voxelize(self.get_mesh(idx), resolution)
        return voxel_grid

    def __len__(self):
        return len(self.scenes)

    def extract_iphone(self, idx):
        scene = ScannetppScene_Release(self.scenes[idx], data_root=self.data_dir)
        extract_rgb(scene)
        extract_masks(scene)
        extract_depth(scene)

    def get_index_from_scene(self, scene_name):
        if isinstance(scene_name, list):
            return [self.scenes.index(name) for name in scene_name]
        return self.scenes.index(scene_name)
    
    def get_voxelized_scene(self, scene_name):
        voxelized_scene_dir = self.get_voxelized_scene_dir(scene_name, resolution=self.data_config.scene_resolution)
        # maybe make this faster by caching loaded scenes
        return torch.load(voxelized_scene_dir / self.get_file_name_voxelized_scene(scene_name))
    
    def get_mesh(self, idx):
        mesh_path = self.data_dir / self.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
        return trimesh.load(mesh_path)

    def __getitem__(self, idx):
        scene_path = self.data_dir / self.scenes[idx]
        camera_path = scene_path / self.camera

        if self.camera == "iphone" and not (
            (camera_path / "rgb").exists()
            and (camera_path / "rgb_masks").exists()
            and (camera_path / "depth").exists()
        ):
            self.extract_iphone(idx)

        if self.camera == "dslr" and not (
            (camera_path / "undistorted_images").exists()
        ):
            raise ValueError("Please run the undistortion script for this scene first")

        mesh = self.get_mesh(idx)

        images_with_params = get_camera_params(scene_path, self.camera, None, 0)

        image_dir = "rgb" if self.camera == "iphone" else "undistorted_images"

        if self.data_config.return_voxelized:
            if not self.check_voxelized_scene_exists(self.scenes[idx]):
                print("Voxelizing scene does not exist yet and is created on the fly!")
                self.prepare_scene(self.scenes[idx])
            return {
                "scene_name": self.scenes[idx],
                "mesh": mesh,
                "path_images": self.data_dir / self.scenes[idx] / self.camera / image_dir,
                "camera_params": images_with_params,
                "voxelized_scene": self.get_voxelized_scene_dir(self.scenes[idx], resolution=self.data_config.scene_resolution),
            }
        else:
            return {
                "scene_name": self.scenes[idx],
                "mesh": mesh,
                "path_images": self.data_dir / self.scenes[idx] / self.camera / image_dir,
                "camera_params": images_with_params,
            }


# config = load_yaml_munch("./utils/config.yaml")



# class SceneDatasetTransformToTorch(nn.Module):
#     def __init__(self):
#         self.tensor = torch.zeros(1)

#     def forward(self, data: dict):
#         points, gt = data["training_data"]
#         image_names, camera_params_list = data["images"]

#         images = torch.stack([read_image(image_name) for image_name in image_names]).to(
#             self.tensor
#         )
#         transformation = torch.stack(
#             [
#                 torch.from_numpy(
#                     camera_params["K"] @ camera_params["T_cw"][:3, :]
#                 ).float()
#                 for camera_params in camera_params_list
#             ]
#         ).to(self.tensor)
#         points = torch.tensor(torch.from_numpy(points).float()).to(self.tensor)
#         gt = torch.tensor(torch.from_numpy(gt).float()).to(self.tensor)
#         return images, transformation, points, gt


# def get_image_to_random_vertice(mesh_path):
#     mesh = pv.read(mesh_path)
#     vertices = mesh.points
#     random_indices = np.random.randint(0, vertices.shape[0])
#     return vertices[random_indices]


# def plot_training_example(data_dict):
#     mesh = data_dict["mesh"]
#     points, gt = data_dict["training_data"]
#     image_names, camera_params_list, P_center = data_dict["images"]

#     # if gt.dtype == 'bool':
#     #     points = points[gt.flatten()]
#     #     gt = gt[gt]

#     visualize_mesh(
#         pv.wrap(mesh),
#         images=image_names,
#         camera_params_list=camera_params_list,
#         heat_values=gt,
#         point_coords=points,
#     )


# def plot_mask(dataset, idx):
#     points = get_mask(dataset.data_dir / dataset.scenes[idx])
#     visualize_mesh(
#         dataset.data_dir / dataset.scenes[idx] / "scans" / "mesh_aligned_0.05.ply",
#         point_coords=points,
#     )


# def plot_occupency_grid(data_dict, resolution=0.02):
#     points, gt = data_dict["training_data"]
#     plot_voxel_grid(points, gt, resolution=resolution, ref_mesh=data_dict["mesh"])


# if __name__ == "__main__":
#     dataset = SceneDataset(
#         camera="iphone",
#         n_points=300000,
#         threshold_occ=0.01,
#         representation="occ",
#         visualize=True,
#         resolution=0.015,
#     )

#     coordinates, occupancy_values = dataset.create_voxel_grid(0)
#     mesh_path = dataset.data_dir / dataset.scenes[0] / "scans" / "mesh_aligned_0.05.ply"
#     # plot_voxel_grid(coordinates, occupancy_values, resolution=dataset.resolution)

#     idx = dataset.get_index_from_scene("8b2c0938d6")
#     # plot_mask(dataset, idx)

#     plot_training_example(dataset[0])
#     # plot_occupency_grid(dataset, resolution=dataset.resolution)
