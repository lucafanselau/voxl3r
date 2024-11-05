import os
from pathlib import Path
import time
import igl
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

import pyvista as pv
import trimesh

from extern.scannetpp.common.scene_release import ScannetppScene_Release
from extern.scannetpp.iphone.prepare_iphone_data import extract_depth, extract_masks, extract_rgb
from utils.data_parsing import get_camera_params, get_vertices_labels, load_yaml_munch
from utils.masking import get_mask, get_structures_unstructured_mesh
from utils.transformations import invert_pose, project_points, quaternion_to_rotation_matrix, undistort_image
from utils.visualize import plot_voxel_grid, visualize_images, visualize_mesh, visualize_mesh_without_vertices

class SceneDataset(Dataset):
    def __init__(self, camera="dslr", data_dir="data", n_points=10000, max_seq_len=20, representation="tdf", threshold_occ=0.0, visualize=False):
        self.camera = camera
        self.data_dir = Path(data_dir)
        self.scenes = ["scannet_demo"] # os.list_dir(data_dir)
        self.n_points = n_points
        self.max_seq_len = max_seq_len
        self.cfg = load_yaml_munch(Path(".") / "utils" / "config.yaml")
        self.representation = representation
        
        if threshold_occ > 0.0:
            self.threshold_occ = threshold_occ
        
        self.visualize = visualize

    def __len__(self):
        return len(self.img_labels)

    def get_images_with_3d_point(self, idx, P_world, image_names = None, tolerance=0.9):           
        c_params = get_camera_params(self.data_dir / self.scenes[idx], self.camera, image_names, self.max_seq_len)

        images_names = []
        pixel_coordinate = []
        camera_params_list = []
        
        for key in c_params.keys():
            image_name = key
            c_dict = c_params[key]
            t_cw, R_cw, K = c_dict['t_cw'], c_dict['R_cw'], c_dict['K']
            width, height = c_dict['width'], c_dict['height']
            
            uvs, _ = project_points(P_world, K, R_cw, t_cw)
            
            if uvs is not None:
                u, v = uvs[:, 0]
                w_min, w_max = (0.5 - tolerance/2) * width, (0.5 + tolerance/2) * width
                h_min, h_max = (0.5 - tolerance/2) * height, (0.5 + tolerance/2) * height
                if w_min <= u < w_max and h_min <= v < h_max:
                      
                    images_names.append(image_name)
                    pixel_coordinate.append([u, v])
                    camera_params_list.append(c_dict)
                    
        return images_names, pixel_coordinate, camera_params_list
    
    def extract_iphone(self, idx):
        scene = ScannetppScene_Release(self.scenes[idx], data_root=Path("./") / 'data')
        extract_rgb(scene)
        extract_masks(scene)
        extract_depth(scene)
        
    def sample_scene(self, idx):
        mesh_path = self.data_dir / self.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
        mesh = trimesh.load(mesh_path)
        
        # divide the mesh into two parts: structured and unstructured
        labels = get_vertices_labels(self.data_dir / self.scenes[idx])
        
        mesh_wo_less_sampling_areas, mesh_less_sampling_areas = get_structures_unstructured_mesh(mesh, self.cfg.less_sampling_areas, labels)
        
        n_surface = int(0.6 * self.n_points)
        n_uniformal =int(0.4 * self.n_points)
        n_surface_structured = int(0.8 * n_surface)
        n_surface_unstructured = n_surface - n_surface_structured
        
        amp_noise = 1e-4
        
        points_surface_structured = mesh_wo_less_sampling_areas.sample(n_surface_structured)
        points_surface_structured = points_surface_structured + amp_noise*np.random.randn(points_surface_structured.shape[0], 3)
                
        points_surface_unstructured = mesh_less_sampling_areas.sample(n_surface_unstructured)
        points_surface_unstructured = points_surface_unstructured + amp_noise*np.random.randn(points_surface_unstructured.shape[0], 3)
        
        points_uniformal = np.random.uniform(low=np.array([-1, -1, -1]), high=mesh.extents + np.array([1, 1, 1]), size=(n_uniformal, 3))
              
        points = np.concatenate([points_surface_structured, points_surface_unstructured, points_uniformal], axis=0)
        
        start = time.time()
        print(f"Started computing distance function for {points.shape[0]} points")
        inside_surface_values = igl.signed_distance(points, mesh.vertices, mesh.faces)
        print(f"Time needed to compute SDF: {time.time() - start}")

        np.savez(self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz", points=points, sdf=inside_surface_values[0])
        
    def sample_chunk(self, idx, image_name, center = np.array([0.0,0.0,1.25]), size = np.array([1.5,1.0,2.0]), visualize=False):
 
        if not (self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz").exists():
            self.sample_scene(idx)

        data = np.load(self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz") 
        points, sdf = data['points'], data['sdf']
        
        c_dict = get_camera_params(self.data_dir / self.scenes[idx], self.camera, image_name, 1)
        transformation = c_dict['T_cw']
        _, _, back_transformation = invert_pose(c_dict['R_cw'], c_dict['t_cw'])
        
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = (transformation @ points.T).T[:, :3]
        
        sdf = sdf[(np.abs(points - center) < size/2).all(axis=1)]
        df = np.abs(sdf)
        
        gt = None
        if self.representation == "tdf":
            df[df > 1] = 1
            gt = np.abs(sdf)
        elif self.representation == "occ":
            gt = np.zeros_like(df)
            gt[self.threshold_occ > df] = 1
            gt[df >= self.threshold_occ] = 0
                
        points = points[(np.abs(points - center) < size/2).all(axis=1)]
        


        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = (back_transformation @ points.T).T[:, :3]
        
        mesh = None

        if visualize:
            path = Path(self.data_dir) / self.scenes[idx]
            mesh_path = os.path.join(path, "scans", "mesh_aligned_0.05.ply")
            mesh = trimesh.load(mesh_path)

            # transform mesh to camera coordinate frame
            mesh.apply_transform(transformation)
        
            # slice the mesh to get the sliced structured and unstructured mesh
            for i, offset in enumerate(size):
                plane_origin, plane_normal = np.zeros(3), np.zeros(3)
                plane_origin[i], plane_normal[i] = offset/2, 1
                mesh = mesh.slice_plane(center-plane_origin, plane_normal)
                mesh = mesh.slice_plane(center+plane_origin, -plane_normal)
            
            mesh.apply_transform(back_transformation)
        
        vec_center =  np.array([*center.flatten(),  1])
        P_center = (back_transformation @ vec_center).flatten()[:3]
        images_names, pixel_coordinate, camera_params_list = self.get_images_with_3d_point(idx, P_center, image_names=image_name, tolerance=0.8)
        
        return (points, gt), (images_names, pixel_coordinate, camera_params_list, P_center), mesh  
        
    def __getitem__(self, idx):
        path_camera = self.data_dir / self.scenes[idx]  / self.camera
        
        # check if the data has already been extracted
        if self.camera == "iphone" and not ((path_camera / "rgb").exists() and (path_camera / "rgb_masks").exists() and (path_camera / "depth").exists()):
                self.extract_iphone(idx)
    
        # sample a random chunk of the scene and return points, sdf values, and images with camera parameters
        image_folder = "images" if self.camera == "dslr" else "rgb"
        image_names = sorted(os.listdir(path_camera / image_folder))
        image_len = len(image_names) if self.camera == "dslr" else len(image_names) // 10
        # generate random number using image_len
        np.random.seed(0)
        image_name = image_names[np.random.randint(0, image_len) if self.camera == "dslr" else np.random.randint(0, image_len) * 10]
        
        gt, images, mesh = self.sample_chunk(idx, image_name, visualize=self.visualize)
        
        return {
            'scan': mesh,
            'training_data': gt,
            'images': images,
        }

            
        
def get_image_to_random_vertice(mesh_path):
    mesh = pv.read(mesh_path)
    np.random.seed(42)
    vertices = mesh.points
    random_indices = np.random.randint(0, vertices.shape[0])
    return vertices[random_indices]

def plot_random_training_example(dataset):    
    data_dict = dataset[0]
    mesh = data_dict['scan']
    points, gt = data_dict['training_data']
    image_names, pixel_coordinates, camera_params_list, P_center = data_dict['images']
    
    images = [(dataset.data_dir/ dataset.scenes[0] / dataset.camera / \
        ('images' if dataset.camera == 'dslr' else 'rgb') / name) for name in image_names]
    
    visualize_mesh(pv.wrap(mesh), images=images, camera_params_list=camera_params_list, heat_values=gt, point_coords=points)
    
def plot_mask(dataset):
    points = get_mask(dataset.data_dir / dataset.scenes[0])
    visualize_mesh(dataset.data_dir / dataset.scenes[0] / "scans" / "mesh_aligned_0.05.ply", point_coords=points)
    
def plot_occupency_grid(dataset):
    data_dict = dataset[0]
    points, gt = data_dict['training_data']    
    plot_voxel_grid(points, gt, resolution=0.01)
    
if __name__ == "__main__":
    dataset = SceneDataset(camera="dslr", n_points=300000, threshold_occ=0.01, representation="occ")
    #plot_mask(dataset)
    #plot_random_training_example(dataset)
    plot_occupency_grid(dataset)
    
    # image_path = dataset.data_dir/ dataset.scenes[0] / dataset.camera / ('images' if dataset.camera == 'dslr' else 'rgb') / image_name
    # visualize_mesh(pv.wrap(mesh), images=images, camera_params_list=camera_params_list, point_coords=P_center, plane_distance=0.1, offsets=[0.025, 0.05])
    # visualize_mesh(pv.wrap(mesh), images=images, camera_params_list=camera_params_list, heat_values=signed_distance_values, point_coords=points)
    
    # visualize_mesh(pv.wrap(mesh), images=[image_path], camera_params_list=[dataset.get_camera_params(0, image_name)], heat_values=signed_distance_values, point_coords=points)
    
    # # visualize th distorted and undistorted images
    # idx = -1
    # visualize_images([data['images'][idx], data['undistorted_images'][idx], data["depth images"][idx].double() / data["depth images"][idx].double().max()])
    # visualize_mesh_without_vertices(data['scan path'], data['labels'], LESS_SAMPLING_AREAS)
    
    
    # mesh_path = data['scan path']
    # P_world = get_image_to_random_vertice(mesh_path)
    
    # # # Get images that see the 3D point and their camera parameters
    # image_names, pixel_coordinates, camera_params_list = dataset.get_images_with_3d_point(idx=0, P_world=P_world, tolerance=0.4, verbose=False)

    # # # Prepare the full paths to the images
    # images = [(dataset.data_dir/ dataset.scenes[0] / dataset.camera / \
    #     ('images' if dataset.camera == 'dslr' else 'rgb') / name) for name in image_names]

    # # Visualize the mesh with images
    # visualize_mesh(mesh_path, images[1:2], camera_params_list[1:2], P_world, plane_distance=0.1, offsets=[0.1, 0.2])