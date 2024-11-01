import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.data_parsing import create_camera_extrinsics_csv, get_camera_params, DATA_PREPROCESSING_DIR, CAMERA_EXTRINSICS_FILENAME
from utils.transformations import invert_pose, project_points, quaternion_to_rotation_matrix
from utils.visualize import visualize_images, visualize_mesh

class SceneDataset(Dataset):
    def __init__(self, camera="dslr", data_dir="data"):
        self.camera = camera
        self.data_dir = data_dir
        self.scenes = ["scannet_demo"] # os.list_dir(data_dir)
        

    def __len__(self):
        return len(self.img_labels)
    
    def get_images_with_3d_point(self, idx, P_world, verbose = True, tolerance=0.9):
        
        camera_extrinsics_file = os.path.join(self.data_dir, self.scenes[idx], self.camera, DATA_PREPROCESSING_DIR, CAMERA_EXTRINSICS_FILENAME)
        
        if os.path.exists(camera_extrinsics_file):
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

    def __getitem__(self, idx):
        data_dict = {}
        path = os.path.join(self.data_dir, self.scenes[idx])    
        
        # camera_params = get_camera_params(os.path.join(self.data_dir, self.scenes[idx], self.camera, "colmap", "cameras.txt"))

        # if self.camera == "dslr":
        #     images = torch.stack([read_image(os.path.join(path, "dslr", "images", file)).permute(1, 2, 0) for file in sorted(os.listdir(os.path.join(path, "dslr", "images")))])
        #     undistorted_images = torch.from_numpy(np.array([cv2.undistort(image.numpy(), camera_params['K'], camera_params['dist_coeffs']) for image in images]))
        
        # data_dict['images'] = images
        # data_dict['undistorted_images'] = undistorted_images
        data_dict['scan path'] = os.path.join(path, "scans", "mesh_aligned_0.05.ply")
            
        return data_dict
    
        
if __name__ == "__main__":
    dataset = SceneDataset()
    
    P_world = np.array([[0], [0], [0]])
    
    # Get images that see the 3D point and their camera parameters
    image_names, pixel_coordinates, camera_params_list = dataset.get_images_with_3d_point(
        idx=0,
        P_world=P_world,
        tolerance=0.4,
    )

    # Prepare the full paths to the images
    images = [
        os.path.join(
            dataset.data_dir,
            dataset.scenes[0],
            dataset.camera,
            'images',
            name
        ) for name in image_names
    ]
    
    camera_params_list = camera_params_list[1:2]
    images = images[1:2]

    # Visualize the mesh with images
    mesh_path = dataset[0]['scan path']
    visualize_mesh(mesh_path, images, camera_params_list, P_world, plane_distance=0.1, offsets=[0.1, 0.2])