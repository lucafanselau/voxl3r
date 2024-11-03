import os
from pathlib import Path

import cv2
import torch
from torchvision.io import read_image

from utils.data_parsing import get_camera_params
import numpy as np
import matplotlib.pyplot as plt

def get_mask(scene_path):
    camera_params = get_camera_params(scene_path, "iphone", None, None)
    
    all_points = []
    for image_name in camera_params.keys():
        c_params = camera_params[image_name]
        K, dist_coeffs, w, h = c_params['K'], c_params['dist_coeffs'], c_params['width'], c_params['height']
        
        depth = read_image(scene_path / "iphone" / "depth" / (image_name.split(".")[0] + ".png")).float()
        depth *= 0.001
        depth = depth.squeeze(0).numpy()
        H, W = depth.shape
        
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (int(w), int(h)), 0, (W, H)
        )
        K_inv = np.linalg.inv(new_camera_matrix)
        
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.astype(float)
        v = v.astype(float)
        ones = np.ones_like(u)
        pixels = np.stack([u, v, ones], axis=0).reshape(3, -1)  # [3, H*W]
        
        points = (K_inv @ pixels * depth.reshape(-1)).reshape(3, -1).T  # [H*W, 3]
        all_points.append(points)
        break
    
    all_points = np.concatenate(all_points, axis=0)
    return all_points

if __name__ == "__main__":
    scene_path = Path("./data/scannet_demo")  # Replace with your actual scene path
    points = get_mask(scene_path)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Point Cloud')
    plt.show()
