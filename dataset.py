from pathlib import Path
import time
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

import pyvista as pv
import trimesh
from torch import nn
from torchvision.io import read_image

import binvox


from extern.scannetpp.common.scene_release import ScannetppScene_Release
from extern.scannetpp.iphone.prepare_iphone_data import (
    extract_depth,
    extract_masks,
    extract_rgb,
)
from utils.data_parsing import get_camera_params, get_image_names_with_extrinsics, get_vertices_labels, load_yaml_munch
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


class SceneDataset(Dataset):
    def __init__(
        self,
        camera="iphone",
        data_dir="datasets/scannetpp/data",
        n_points=200000,
        max_seq_len=20,
        representation="tdf",
        threshold_occ=0.0,
        visualize=False,
        resolution=0.02,
        seed=42,
        scenes=None,
    ):
        self.camera = camera
        self.data_dir = Path(data_dir)
        self.scenes = [x.name for x in self.data_dir.glob("*") if x.is_dir()] if scenes is None else scenes
        self.n_points = n_points
        self.max_seq_len = max_seq_len
        self.cfg = load_yaml_munch(Path(".") / "utils" / "config.yaml")
        self.representation = representation
        
        self.resolution = resolution
        
        self.seed = seed

        if threshold_occ > 0.0:
            self.threshold_occ = threshold_occ

        self.visualize = visualize

    def __len__(self):
        return len(self.scenes)

    def get_images_with_3d_point(self, idx, points, image_names=None, tolerance=0.9):
        c_params = get_camera_params(
            self.data_dir / self.scenes[idx], self.camera, image_names, self.max_seq_len
        )

        images_names = []
        pixel_coordinate = []
        camera_params_list = []

        uvs_dict, _ = project_image_plane(c_params, points)

        for key in c_params.keys():
            c_dict = c_params[key]

            if uvs_dict[key] is not None:
                u, v = uvs_dict[key][:, 0]
                w_min, w_max = (0.5 - tolerance / 2) * c_dict["width"], (
                    0.5 + tolerance / 2
                ) * c_dict["width"]
                h_min, h_max = (0.5 - tolerance / 2) * c_dict["height"], (
                    0.5 + tolerance / 2
                ) * c_dict["height"]
                if w_min <= u < w_max and h_min <= v < h_max:

                    images_names.append(key)
                    pixel_coordinate.append([u, v])
                    camera_params_list.append(c_dict)

        return images_names, pixel_coordinate, camera_params_list

    def extract_iphone(self, idx):
        scene = ScannetppScene_Release(self.scenes[idx], data_root=self.data_dir)
        extract_rgb(scene)
        extract_masks(scene)
        extract_depth(scene)
        
    def voxel_grid_to_point_grid(self, voxel_grid, even_distribution=True):
        occupancy_grid = voxel_grid.encoding.dense
        indices = np.indices(occupancy_grid.shape).reshape(3, -1).T
        origin = voxel_grid.bounds[0]
        coordinates = origin + (indices + 0.5) * self.resolution
        occupancy_values = occupancy_grid.flatten()
        
        if even_distribution:
            false_indices = np.where(~occupancy_values)[0]
            true_indices = np.where(occupancy_values)[0]
            false_indices = np.random.choice(false_indices, true_indices.shape[0], replace=False)
            occupancy_values = np.concatenate([occupancy_values[true_indices], occupancy_values[false_indices]])
            coordinates = np.concatenate([coordinates[true_indices], coordinates[false_indices]])
            
        return coordinates, occupancy_values
        
        
    def create_voxel_grid(self, idx, even_distribution=True):
        
        if (self.data_dir / self.scenes[idx] / "scans" / f"occ_res_{self.resolution}.npz").exists():
            data = np.load(self.data_dir / self.scenes[idx] / "scans" / f"occ_res_{self.resolution}.npz")
            return data["coordinates"], data["occupancy_values"]
        
        mesh_path = self.data_dir / self.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
        mesh = trimesh.load(mesh_path)
        voxel_grid = mesh.voxelized(self.resolution)
        coordinates, occupancy_values = self.voxel_grid_to_point_grid(voxel_grid, even_distribution)
            
        return coordinates, occupancy_values
    
    def sample_scene(self, idx):
        mesh_path = self.data_dir / self.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
        mesh = trimesh.load(mesh_path)

        # divide the mesh into two parts: structured and unstructured
        labels = get_vertices_labels(self.data_dir / self.scenes[idx])

        mesh_wo_less_sampling_areas, mesh_less_sampling_areas = (
            get_structures_unstructured_mesh(mesh, self.cfg.less_sampling_areas, labels)
        )

        n_surface = int(0.6 * self.n_points)
        n_uniformal = int(0.4 * self.n_points)
        n_surface_structured = int(0.8 * n_surface)
        n_surface_unstructured = n_surface - n_surface_structured

        amp_noise = 1e-4
        points_surface_structured = mesh_wo_less_sampling_areas.sample(
            n_surface_structured
        )
        points_surface_structured = (
            points_surface_structured
            + amp_noise * np.random.randn(points_surface_structured.shape[0], 3)
        )

        points_surface_unstructured = mesh_less_sampling_areas.sample(
            n_surface_unstructured
        )
        points_surface_unstructured = (
            points_surface_unstructured
            + amp_noise * np.random.randn(points_surface_unstructured.shape[0], 3)
        )

        points_uniformal = np.random.uniform(
            low=np.array([-1, -1, -1]),
            high=mesh.extents + np.array([1, 1, 1]),
            size=(n_uniformal, 3),
        )

        points = np.concatenate(
            [points_surface_structured, points_surface_unstructured, points_uniformal],
            axis=0,
        )

        start = time.time()
        print(f"Started computing distance function for {points.shape[0]} points")
        if self.cfg.sdf_strategy == "igl":
            import igl

            inside_surface_values = igl.signed_distance(
                points, mesh.vertices, mesh.faces
            )[0]
        elif self.cfg.sdf_strategy == "pysdf":
            from pysdf import SDF

            f = SDF(mesh.vertices, mesh.faces)
            inside_surface_values = f(points)
        print(f"Time needed to compute SDF: {time.time() - start}")

        np.savez(
            self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz",
            points=points,
            sdf=inside_surface_values,
        )
        
    def chunk_whole_scene(self, idx):
        path_camera = self.data_dir / self.scenes[idx] / self.camera
        image_names = get_image_names_with_extrinsics(path_camera)
        len_chunk = len(image_names) // self.max_seq_len
        chunks = []
        for i in range(len_chunk - 1):
            image_name = image_names[i * self.max_seq_len]
            
            training, images, mesh = self.sample_chunk(idx, image_name)
            chunks.append({
                "mesh": mesh,
                "training_data": training,
                "images": images,
            })
        return chunks

    def sample_chunk(
        self,
        idx,
        image_name,
        center=np.array([0.0, 0.0, 1.25]),
        size=np.array([1.5, 1.0, 2.0]),
        visualize=False,
    ):  
        if self.representation == "tdf":
            if not (
                self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz"
            ).exists():
                self.sample_scene(idx)

            data = np.load(
                self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz"
            )
            points, sdf = data["points"], data["sdf"]
            gt = np.abs(sdf)
            gt[gt > 1] = 1
            
        elif self.representation == "occ":
            points, gt = self.create_voxel_grid(idx)
    

        c_dict = get_camera_params(
            self.data_dir / self.scenes[idx], self.camera, image_name, 1
        )[image_name]
        transformation = c_dict["T_cw"]
        _, _, back_transformation = invert_pose(c_dict["R_cw"], c_dict["t_cw"])

        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = (transformation @ points.T).T[:, :3]
        
        gt = gt[(np.abs(points - center) < size / 2).all(axis=1)]
        points = points[(np.abs(points - center) < size / 2).all(axis=1)]
        
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = (back_transformation @ points.T).T[:, :3]
        
        mesh = None

        if visualize:
            path = Path(self.data_dir) / self.scenes[idx]
            mesh_path = path / "scans" / "mesh_aligned_0.05.ply"
            mesh = trimesh.load(mesh_path)

            # transform mesh to camera coordinate frame
            mesh.apply_transform(transformation)

            for i, offset in enumerate(size):
                plane_origin, plane_normal = np.zeros(3), np.zeros(3)
                plane_origin[i], plane_normal[i] = offset / 2, 1
                mesh = mesh.slice_plane(center - plane_origin, plane_normal)
                mesh = mesh.slice_plane(center + plane_origin, -plane_normal)

            mesh.apply_transform(back_transformation)

        vec_center = np.array([*center.flatten(), 1])
        P_center = (back_transformation @ vec_center).flatten()[:3]
        image_names, pixel_coordinate, camera_params_list = (
            self.get_images_with_3d_point(
                idx, P_center, image_names=image_name, tolerance=0.8
            )
        )

        image_file = "rgb" if self.camera == "iphone" else "images"
        image_names = [
            self.data_dir / self.scenes[idx] / self.camera / image_file / image_name
            for image_name in image_names
        ]
        

        return (points, gt), (image_names, camera_params_list, P_center), mesh

    def get_index_from_scene(self, scene_name):
        if isinstance(scene_name, list):
            return [self.scenes.index(name) for name in scene_name]
        return self.scenes.index(scene_name)

    def __getitem__(self, idx):
        path_camera = self.data_dir / self.scenes[idx] / self.camera

        # check if the data has already been extracted
        if self.camera == "iphone" and not (
            (path_camera / "rgb").exists()
            and (path_camera / "rgb_masks").exists()
            and (path_camera / "depth").exists()
        ):
            self.extract_iphone(idx)

        # sample a random chunk of the scene and return points, sdf values, and images with camera parameters
        image_names = get_image_names_with_extrinsics(path_camera)
        image_name = image_names[np.random.randint(0, len(image_names))]

        np.random.seed(self.seed)
        training, images, mesh = self.sample_chunk(
            idx, image_name, visualize=self.visualize
        )

        return {
            "mesh": mesh,
            "training_data": training,
            "images": images,
        }


class SceneDatasetTransformToTorch:
    def __init__(self, target_device):
        self.target_device = target_device

    def forward(self, data: dict):
        points, gt = data["training_data"]
        image_names, camera_params_list, _ = data["images"]

        images = torch.stack([read_image(image_name) for image_name in image_names]).to(
            self.target_device
        )
        transformation = torch.stack(
            [
                torch.from_numpy(
                    camera_params["K"] @ camera_params["T_cw"][:3, :]
                ).float()
                for camera_params in camera_params_list
            ]
        ).to(self.target_device)
        points = torch.tensor(torch.from_numpy(points).float()).to(self.target_device)
        gt = torch.tensor(torch.from_numpy(gt).float()).to(self.target_device)
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
        resolution = 0.015
    )
    
    coordinates, occupancy_values = dataset.create_voxel_grid(0)
    mesh_path = dataset.data_dir / dataset.scenes[0] / "scans" / "mesh_aligned_0.05.ply"
    # plot_voxel_grid(coordinates, occupancy_values, resolution=dataset.resolution)

    idx = dataset.get_index_from_scene("8b2c0938d6")
    #plot_mask(dataset, idx)
    
    plot_training_example(dataset[0])
    #plot_occupency_grid(dataset, resolution=dataset.resolution)
