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
from utils.data_parsing import create_camera_extrinsics_csv, get_camera_intrisics, DATA_PREPROCESSING_DIR, CAMERA_EXTRINSICS_FILENAME, read_depth_bin
from utils.transformations import invert_pose, project_points, quaternion_to_rotation_matrix, undistort_image
from utils.visualize import visualize_images, visualize_mesh, visualize_mesh_without_vertices

LESS_SAMPLING_AREAS = ['wall', 'floor', 'ceiling']

class SceneDataset(Dataset):
    def __init__(self, camera="dslr", data_dir="data", n_points=10000):
        self.camera = camera
        self.data_dir = Path(data_dir)
        self.scenes = ["scannet_demo"] # os.list_dir(data_dir)
        self.n_points = n_points
        

    def __len__(self):
        return len(self.img_labels)
    
    def get_camera_params(self, idx, image_name=None): 
        camera_path = self.data_dir / self.scenes[idx] / self.camera
        camera_extrinsics_file = camera_path / DATA_PREPROCESSING_DIR / CAMERA_EXTRINSICS_FILENAME

        camera_intrinsics = get_camera_intrisics(camera_path / "colmap" / "cameras.txt")
        
        # Check if the camera extrinsics file exists; if not, create it
        if not camera_extrinsics_file.exists():
            create_camera_extrinsics_csv(
                self.scenes[idx], camera=self.camera, data_dir=self.data_dir
            ) 

        df = pd.read_csv(camera_extrinsics_file)

        if image_name is not None:
            df_image = df[df['NAME'] == image_name]
            if df_image.empty:
                raise ValueError(f"No camera extrinsics found for image {image_name}")
            row = df_image.iloc[0]

            qw, qx, qy, qz = row[['QW', 'QX', 'QY', 'QZ']].values
            tx, ty, tz = row[['TX', 'TY', 'TZ']].values
            t_cw = np.array([[tx], [ty], [tz]])
            R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)

            return {
                    'R_cw': R_cw, 
                    't_cw': t_cw,
                    'K': camera_intrinsics['K'],
                    'width': camera_intrinsics['width'],
                    'height': camera_intrinsics['height'],
                    }
        else:
            extrinsics_dict = {}
            for _, row in df.iterrows():
                name = row['NAME']
                qw, qx, qy, qz = row[['QW', 'QX', 'QY', 'QZ']].values
                tx, ty, tz = row[['TX', 'TY', 'TZ']].values
                t_cw = np.array([[tx], [ty], [tz]])
                R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)
                extrinsics_dict[name] = {
                    'R_cw': R_cw, 
                    't_cw': t_cw,
                    'K': camera_intrinsics['K'],
                    'width': camera_intrinsics['width'],
                    'height': camera_intrinsics['height'],
                    }
            return extrinsics_dict

    
    def get_images_with_3d_point(self, idx, P_world, verbose = True, tolerance=0.9):            
        c_params= self.get_camera_params(idx)

        images_names = []
        pixel_coordinate = []
        camera_params_list = []
        
        for key in c_params.keys():
            image_name = key
            c_dict = c_params[key]
            t_cw = c_dict['t_cw']
            R_cw = c_dict['R_cw']
            K = c_dict['K']
            width = c_dict['width']
            height = c_dict['height']
            
            uvs, _ = project_points(P_world, K, R_cw, t_cw)
            
            if uvs is not None:
                u, v = uvs[:, 0]
                w_min, w_max = (0.5 - tolerance/2) * width, (0.5 + tolerance/2) * width
                h_min, h_max = (0.5 - tolerance/2) * height, (0.5 + tolerance/2) * height
                if w_min <= u < w_max and h_min <= v < h_max:
                      
                    images_names.append(image_name)
                    pixel_coordinate.append([u, v])
                    camera_params_list.append(c_dict)

                    if verbose:
                        print(f"Point is visible in image {image_name} at pixel coordinates ({u:.2f}, {v:.2f})")                        
                else:
                    if verbose:
                        print(f"Point projects outside image {image_name}")
            else:
                if verbose:
                    print(f"Point is behind the camera in image {image_name}")
                    
        return images_names, pixel_coordinate, camera_params_list
    
    def extract_iphone(self, idx):
        scene = ScannetppScene_Release(self.scenes[idx], data_root=Path("./") / 'data')
        extract_rgb(scene)
        extract_masks(scene)
        extract_depth(scene)
        
    def sample_chunk(self, idx, n_points, labels, image_name, center = np.array([0.0,0.0,1.0]), size = np.array([1.,1.,1.]), backproject=False):
        path = Path(self.data_dir) / self.scenes[idx]
        mesh_path = os.path.join(path, "scans", "mesh_aligned_0.05.ply")
        mesh = trimesh.load(mesh_path, process=False)
        
        # transform mesh to camera coordinate frame

        c_dict = self.get_camera_params(idx, image_name=image_name)
        R_cw, t_cw = c_dict['R_cw'], c_dict['t_cw']
        transformation = np.eye(4)
        transformation[:3, :3] = R_cw
        transformation[:3, 3] = t_cw.flatten()
        mesh.apply_transform(transformation)
        
         # divide the mesh into two parts: structured and unstructured
        labels = self.get_vertices_labels(idx)
        mesh_wo_unstructured = mesh.copy()
        mesh_w_unstructured = mesh.copy()
        
        vertices_to_remove = set()
        for label in LESS_SAMPLING_AREAS:
            if label in labels:
                vertices_to_remove.update(labels [label])
                        
        mask = np.ones(mesh.vertices.shape[0], dtype=bool)
        mask[list(vertices_to_remove)] = False
        face_mask = mask[mesh.faces].any(axis=1)
        inverted_face_mask = ~face_mask
        mesh_wo_unstructured.update_faces(face_mask)
        mesh_w_unstructured.update_faces(inverted_face_mask)
        
        # slice the mesh to get the sliced structured and unstructured mesh
        for i, offset in enumerate(size):
            plane_origin, plane_normal = np.zeros(3), np.zeros(3)
            plane_origin[i], plane_normal[i] = offset/2, 1
            mesh_wo_unstructured = mesh_wo_unstructured.slice_plane(center-plane_origin, plane_normal)
            mesh_wo_unstructured = mesh_wo_unstructured.slice_plane(center+plane_origin, -plane_normal)
            mesh_w_unstructured = mesh_w_unstructured.slice_plane(center-plane_origin, plane_normal)
            mesh_w_unstructured = mesh_w_unstructured.slice_plane(center+plane_origin, -plane_normal)
            
        n_surface = int(0.8 * n_points)
        n_uniformal =int(0.2 * n_points)
        n_surface_structured = int(0.8 * n_surface)
        n_surface_unstructured = n_surface - n_surface_structured
        
        amp_noise = 0.01
        
        points_surface_structured = mesh_wo_unstructured.sample(n_surface_structured)
        points_surface_structured = points_surface_structured + amp_noise*np.random.randn(points_surface_structured.shape[0], 3)
        points_surface_structured = np.delete(points_surface_structured, ((np.abs(points_surface_structured - center) > size[2]/2).any(axis=1)), axis=0)
        
        points_surface_unstructured = mesh_w_unstructured.sample(n_surface_unstructured)
        points_surface_unstructured = points_surface_unstructured + amp_noise*np.random.randn(points_surface_unstructured.shape[0], 3)
        points_surface_unstructured = np.delete(points_surface_unstructured, ((np.abs(points_surface_unstructured - center) > size[2]/2).any(axis=1)), axis=0)
        
        cube_min = np.array(center + size/2)
        cube_max = np.array(center - size/2)
        points_uniformal = np.random.uniform(low=cube_min, high=cube_max, size=(n_uniformal, 3))
        
        points = np.concatenate([points_surface_structured, points_surface_unstructured, points_uniformal], axis=0)
        
        start = time.time()
        inside_surface_values = igl.signed_distance(points, mesh.vertices, mesh.faces)
        print(f"Time needed to compute SDF: {time.time() - start}")

        # transform points to world coordinate frame
        mesh = trimesh.util.concatenate(mesh_wo_unstructured, mesh_w_unstructured)
        if backproject:
            R_wc, t_wc = invert_pose(R_cw, t_cw)
            transformation[:3, :3] = R_wc
            transformation[:3, 3] = t_wc.flatten()
            mesh.apply_transform(transformation)
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            points = (transformation @ points.T).T[:, :3]
        
        return mesh, points, inside_surface_values[0]
        
    
    def get_vertices_labels(self, idx):
        path = Path(self.data_dir) / self.scenes[idx]
        with open(os.path.join(path, "scans", "segments_anno.json"), 'r') as file:
            annotations = json.load(file)    
        labels= {}
        for object in annotations['segGroups']:
            if object['label'] not in labels.keys():
                labels[object['label']] = object['segments']
            else:
                labels[object['label']].extend(object['segments'])
        return labels
    
        
    def __getitem__(self, idx):
        path = Path(self.data_dir) / self.scenes[idx]
        path_camera = path / self.camera
        # check if the data has already been extracted
        if self.camera == "iphone" and not ((path_camera / "rgb").exists() and (path_camera / "rgb_masks").exists() and (path_camera / "depth").exists()):
                self.extract_iphone(idx)
        
        data_dict = {}
        
        data_dict['labels'] = self.get_vertices_labels(idx)
        
        camera_params = get_camera_intrisics(path_camera / "colmap" / "cameras.txt")
        
        image_folder = "images" if self.camera == "dslr" else "rgb"
        data_dict['images'] = torch.stack([read_image(path_camera / image_folder / file).permute(1, 2, 0) for file in sorted(os.listdir(path_camera / image_folder)[:100])])
        data_dict['undistorted_images'] = torch.from_numpy(np.array([undistort_image(image.numpy(), camera_params['K'], camera_params['dist_coeffs']) for image in data_dict['images']]))

        if self.camera == "iphone":
            data_dict['depth images'] = torch.stack([read_image(path_camera /  "depth" / file).permute(1, 2, 0) for file in sorted(os.listdir(path_camera / "depth")[:100])])
        
        data_dict['scan path'] = path / "scans" / "mesh_aligned_0.05.ply"
        
        
        return data_dict
        
def get_image_to_random_vertice(mesh_path):
    mesh = pv.read(mesh_path)
    np.random.seed(42)
    vertices = mesh.points
    random_indices = np.random.randint(0, vertices.shape[0])
    return vertices[random_indices]
        
if __name__ == "__main__":
    dataset = SceneDataset(camera="iphone")
    data = dataset[0]
    image_name = "frame_000000.jpg"
    mesh, points, signed_distance_values = dataset.sample_chunk(0, 10000, data['labels'], image_name, backproject=True)

    image_path = dataset.data_dir/ dataset.scenes[0] / dataset.camera / ('images' if dataset.camera == 'dslr' else 'rgb') / image_name
    visualize_mesh(pv.wrap(mesh), images=[image_path], camera_params_list=[dataset.get_camera_params(0, image_name)], heat_values=signed_distance_values, point_coords=points)
    
    # visualize th distorted and undistorted images
    idx = -1
    #visualize_images([data['images'][idx], data['undistorted_images'][idx], data["depth images"][idx].double() / data["depth images"][idx].double().max()])
    #visualize_mesh_without_vertices(data['scan path'], data['labels'], LESS_SAMPLING_AREAS)
    
    
    mesh_path = data['scan path']
    P_world = get_image_to_random_vertice(mesh_path)
    
    # # Get images that see the 3D point and their camera parameters
    image_names, pixel_coordinates, camera_params_list = dataset.get_images_with_3d_point(idx=0, P_world=P_world, tolerance=0.4, verbose=False)

    # # Prepare the full paths to the images
    images = [(dataset.data_dir/ dataset.scenes[0] / dataset.camera / \
        ('images' if dataset.camera == 'dslr' else 'rgb') / name) for name in image_names]

    # Visualize the mesh with images
    visualize_mesh(mesh_path, images[1:2], camera_params_list[1:2], P_world, plane_distance=0.1, offsets=[0.1, 0.2])