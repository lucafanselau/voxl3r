import os
from pathlib import Path
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

import trimesh

from extern.scannetpp.common.scene_release import ScannetppScene_Release
from extern.scannetpp.iphone.prepare_iphone_data import extract_depth, extract_masks, extract_rgb
from utils.data_parsing import create_camera_extrinsics_csv, get_camera_params, DATA_PREPROCESSING_DIR, CAMERA_EXTRINSICS_FILENAME, read_depth_bin
from utils.transformations import invert_pose, project_points, quaternion_to_rotation_matrix, undistort_image
from utils.visualize import visualize_images, visualize_mesh, visualize_mesh_without_vertices

LESS_SAMPLING_AREAS = ['wall', 'floor', 'ceiling']

class SceneDataset(Dataset):
    def __init__(self, camera="dslr", data_dir="data"):
        self.camera = camera
        self.data_dir = Path(data_dir)
        self.scenes = ["scannet_demo"] # os.list_dir(data_dir)
        

    def __len__(self):
        return len(self.img_labels)
    
    def get_images_with_3d_point(self, idx, P_world, verbose = True, tolerance=0.9):
        
        camera_extrinsics_file = self.data_dir / self.scenes[idx] / self.camera / DATA_PREPROCESSING_DIR / CAMERA_EXTRINSICS_FILENAME
        
        if camera_extrinsics_file.exists():
            df = pd.read_csv(camera_extrinsics_file)
        else:
            create_camera_extrinsics_csv(self.scenes[idx], camera=self.camera, data_dir=self.data_dir)
            df = pd.read_csv(camera_extrinsics_file)
            
        camera_params = get_camera_params(os.path.join(self.data_dir, self.scenes[idx], self.camera, "colmap", "cameras.txt"))

        images_names = []
        pixel_coordinate = []
        camera_params_list = []
        
        for _, row in df.iterrows():
            # Extract data for the current image
            image_name = row['NAME']
            qw, qx, qy, qz = row[['QW', 'QX', 'QY', 'QZ']].values
            tx, ty, tz = row[['TX', 'TY', 'TZ']].values
            t_cw = np.array([[tx], [ty], [tz]])

            # Convert quaternion to rotation matrix
            R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            
            # Project the point
            uvs, _ = project_points(P_world, camera_params["K"], R_cw, t_cw)
            
            if uvs is not None:
                u, v = uvs[:, 0]
                w_min, w_max = (0.5 - tolerance/2) * camera_params["width"], (0.5 + tolerance/2) * camera_params["width"]
                h_min, h_max = (0.5 - tolerance/2) * camera_params["height"], (0.5 + tolerance/2) * camera_params["height"]
                if w_min <= u < w_max and h_min <= v < h_max:
                    
                    
                    cam_params = {
                    'R_cw': R_cw,
                    't_cw': t_cw,
                    'K': camera_params['K'],
                    'height': camera_params['height'],
                    'width': camera_params['width'],
                    }
                    
                    images_names.append(image_name)
                    pixel_coordinate.append([u, v])
                    camera_params_list.append(cam_params)

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
        
    def __getitem__(self, idx):
        path = Path(self.data_dir) / self.scenes[idx]
        path_camera = path / self.camera
        # check if the data has already been extracted
        if self.camera == "iphone" and not ((path_camera / "rgb").exists() and (path_camera / "rgb_masks").exists() and (path_camera / "depth").exists()):
                self.extract_iphone(idx)
        
        data_dict = {}
        
        # Open and read the JSON file
        with open(os.path.join(path, "scans", "segments_anno.json"), 'r') as file:
            annotations = json.load(file) 
            
        data_dict['labels'] = {}
            
        for object in annotations['segGroups']:
            if object['label'] not in data_dict['labels'].keys():
                data_dict['labels'][object['label']] = object['segments']
            else:
                data_dict['labels'][object['label']].extend(object['segments'])

        camera_params = get_camera_params(path_camera / "colmap" / "cameras.txt")
        
        image_folder = "images" if self.camera == "dslr" else "rgb"
        data_dict['images'] = torch.stack([read_image(path_camera / image_folder / file).permute(1, 2, 0) for file in sorted(os.listdir(path_camera / image_folder)[:100])])
        data_dict['undistorted_images'] = torch.from_numpy(np.array([undistort_image(image.numpy(), camera_params['K'], camera_params['dist_coeffs']) for image in data_dict['images']]))

        if self.camera == "iphone":
            data_dict['depth images'] = torch.stack([read_image(path_camera /  "depth" / file).permute(1, 2, 0) for file in sorted(os.listdir(path_camera / "depth")[:100])])
        
        data_dict['scan path'] = os.path.join(path, "scans", "mesh_aligned_0.05.ply")
            
        return data_dict

        
if __name__ == "__main__":
    dataset = SceneDataset(camera="iphone")
    data = dataset[0]
    
    # visualize th distorted and undistorted images
    idx = -1
    #visualize_images([data['images'][idx], data['undistorted_images'][idx], data["depth images"][idx].double() / data["depth images"][idx].double().max()])
    #visualize_mesh_without_vertices(dataset[0]['scan path'], dataset[0]['labels'], LESS_SAMPLING_AREAS)
    
    
    mesh_path = dataset[0]['scan path']
    # Load the mesh from a file
    mesh = trimesh.load(mesh_path)
    
    # Set a random seed for reproducibility (optional)
    np.random.seed(12)

    # Generate a random index
    vertices = mesh.vertices
    random_index = np.random.randint(0, vertices.shape[0])

    # Get the coordinates of the random vertex
    P_world = vertices[random_index]

    print(f"Random Vertex Index: {random_index}")
    print(f"Coordinates: {P_world}")
    
    # Get images that see the 3D point and their camera parameters
    image_names, pixel_coordinates, camera_params_list = dataset.get_images_with_3d_point(
        idx=0,
        P_world=P_world,
        tolerance=0.4,
        verbose=False
    )

    # Prepare the full paths to the images
    images = [
        os.path.join(
            dataset.data_dir,
            dataset.scenes[0],
            dataset.camera,
            'rgb',
            name
        ) for name in image_names
    ]
    
    camera_params_list = camera_params_list[1:2]
    images = images[1:2]

    # Visualize the mesh with images
    visualize_mesh(mesh_path, images, camera_params_list, P_world, plane_distance=0.1, offsets=[0.1, 0.2])