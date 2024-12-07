from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional
from git import List
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

import pyvista as pv
import trimesh
from torch import nn
from torchvision.io import read_image


from extern.scannetpp.common.scene_release import ScannetppScene_Release
from extern.scannetpp.iphone.prepare_iphone_data import (
    extract_depth,
    extract_masks,
    extract_rgb,
)
from utils.data_parsing import (
    get_camera_params,
    get_image_names_with_extrinsics,
    get_vertices_labels,
    load_yaml_munch,
)
from utils.masking import get_mask, get_structures_unstructured_mesh
from utils.transformations import (
    invert_pose,
    project_image_plane,
    quaternion_to_rotation_matrix,
    undistort_image,
)
from utils.visualize import (
    plot_voxel_grid,
    visualize_mesh,
    visualize_mesh_without_vertices,
)


@dataclass
class SceneDatasetConfig:
    data_dir: str = "datasets/scannetpp/data"
    storage_preprocessing: str = "prepared_grids"
    camera: str = "dslr"
    scenes: Optional[List[str]] = None


class SceneDataset(Dataset):
    def __init__(
        self,
        data_config: SceneDatasetConfig,
    ):
        self.data_config = data_config
        self.camera = data_config.camera
        self.data_dir = Path(data_config.data_dir)
        self.scenes = (
            [x.name for x in self.data_dir.glob("*") if x.is_dir()]
            if data_config.scenes is None
            else data_config.scenes
        )

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

        mesh_path = self.data_dir / self.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
        mesh = trimesh.load(mesh_path)

        images_with_params = get_camera_params(scene_path, self.camera, None, 0)

        image_dir = "rgb" if self.camera == "iphone" else "undistorted_images"

        return {
            "scene_name": self.scenes[idx],
            "mesh": mesh,
            "path_images": self.data_dir / self.scenes[idx] / self.camera / image_dir,
            "camera_params": images_with_params,
        }

config = load_yaml_munch("./utils/config.yaml")

class SceneDatasetTransformLoadImages(nn.Module):
    def __init__(self):
        self.tensor = torch.zeros(1)

    def forward(self, data: dict, images_loaded = False):
        """
        data dict is the data dict returned by create_chunk
        """
        camera_params = data["camera_params"]
        
        if images_loaded:
            images = data["images"]
        else:
            images_dir = data["image_names"]
            images_dir = [str(Path(config.data_dir) / Path(*Path(image_name).parts[Path(image_name).parts.index("data") + 3 :])) for image_name in images_dir]
        
            images = torch.stack([read_image(image_dir) for image_dir in images_dir]).to(
                self.tensor
            )
        T_cw = torch.stack(
            [
               cp["T_cw"] if torch.is_tensor(cp["T_cw"]) else torch.from_numpy(cp["T_cw"]).float()
                for cp in camera_params.values()
            ]
        ).to(self.tensor)
        transformation = torch.stack(
            [
                cp["K"] @ cp["T_cw"][:3, :] if torch.is_tensor(cp["T_cw"]) else torch.from_numpy(
                    cp["K"] @ cp["T_cw"][:3, :]
                ).float()
                for cp in camera_params.values()
            ]
        ).to(self.tensor)
        return images, transformation, T_cw


class SceneDatasetTransformToTorch(nn.Module):
    def __init__(self):
        self.tensor = torch.zeros(1)

    def forward(self, data: dict):
        points, gt = data["training_data"]
        image_names, camera_params_list = data["images"]

        images = torch.stack([read_image(image_name) for image_name in image_names]).to(
            self.tensor
        )
        transformation = torch.stack(
            [
                torch.from_numpy(
                    camera_params["K"] @ camera_params["T_cw"][:3, :]
                ).float()
                for camera_params in camera_params_list
            ]
        ).to(self.tensor)
        points = torch.tensor(torch.from_numpy(points).float()).to(self.tensor)
        gt = torch.tensor(torch.from_numpy(gt).float()).to(self.tensor)
        return images, transformation, points, gt


def get_image_to_random_vertice(mesh_path):
    mesh = pv.read(mesh_path)
    vertices = mesh.points
    random_indices = np.random.randint(0, vertices.shape[0])
    return vertices[random_indices]


def plot_training_example(data_dict):
    mesh = data_dict["mesh"]
    points, gt = data_dict["training_data"]
    image_names, camera_params_list, P_center = data_dict["images"]

    # if gt.dtype == 'bool':
    #     points = points[gt.flatten()]
    #     gt = gt[gt]

    visualize_mesh(
        pv.wrap(mesh),
        images=image_names,
        camera_params_list=camera_params_list,
        heat_values=gt,
        point_coords=points,
    )


def plot_mask(dataset, idx):
    points = get_mask(dataset.data_dir / dataset.scenes[idx])
    visualize_mesh(
        dataset.data_dir / dataset.scenes[idx] / "scans" / "mesh_aligned_0.05.ply",
        point_coords=points,
    )


def plot_occupency_grid(data_dict, resolution=0.02):
    points, gt = data_dict["training_data"]
    plot_voxel_grid(points, gt, resolution=resolution, ref_mesh=data_dict["mesh"])


if __name__ == "__main__":
    dataset = SceneDataset(
        camera="iphone",
        n_points=300000,
        threshold_occ=0.01,
        representation="occ",
        visualize=True,
        resolution=0.015,
    )

    coordinates, occupancy_values = dataset.create_voxel_grid(0)
    mesh_path = dataset.data_dir / dataset.scenes[0] / "scans" / "mesh_aligned_0.05.ply"
    # plot_voxel_grid(coordinates, occupancy_values, resolution=dataset.resolution)

    idx = dataset.get_index_from_scene("8b2c0938d6")
    # plot_mask(dataset, idx)

    plot_training_example(dataset[0])
    # plot_occupency_grid(dataset, resolution=dataset.resolution)
