import os
from pathlib import Path
import time

import cv2
import igl
from torchvision.io import read_image
import trimesh

from utils.data_parsing import get_camera_params, get_vertices_labels, load_yaml_munch
import numpy as np
import matplotlib.pyplot as plt

from utils.transformations import invert_pose


def get_structures_unstructured_mesh(mesh, less_sampling_areas, labels):
    mesh_wo_less_sampling_areas = mesh.copy()
    mesh_less_sampling_areas= mesh.copy()
    
    vertices_to_remove = set()
    for label in less_sampling_areas:
        if label in labels:
            vertices_to_remove.update(labels [label])
                    
    mask = np.ones(mesh.vertices.shape[0], dtype=bool)
    mask[list(vertices_to_remove)] = False
    face_mask = mask[mesh.faces].any(axis=1)
    inverted_face_mask = ~face_mask
    mesh_wo_less_sampling_areas.update_faces(face_mask)
    mesh_less_sampling_areas.update_faces(inverted_face_mask)
    
    return mesh_wo_less_sampling_areas, mesh_less_sampling_areas


def get_mask(scene_path, unstructured_tolerance = 0.20, structured_tolerance = 0.08, max_distance = 0.5):
    camera_params = get_camera_params(scene_path, "iphone", None, None)
    
    mesh = trimesh.load(scene_path / "scans" / "mesh_aligned_0.05.ply")
    cfg = load_yaml_munch(Path("utils") / "config.yaml")
    mesh_wo_less_sampling_areas, mesh_less_sampling_areas = get_structures_unstructured_mesh(mesh, cfg.less_sampling_areas, get_vertices_labels(scene_path))
    
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
        # transform points to world coordinates
        R_cw, t_cw = c_params['R_cw'], c_params['t_cw']
        R_wc, t_wc, T_wc = invert_pose(R_cw, t_cw)
        points = (R_wc @ points.T + t_wc).T
        indices = np.random.choice(points.shape[0], 500)
        all_points.append(points[indices])
        
    all_points = np.concatenate(all_points, axis=0)
    
    return all_points
    
    # remove points from structured regions
    start = time.time()
    print(f"Started computing distance function for {all_points.shape[0]} points")
    inside_surface_values = np.abs(igl.signed_distance(all_points, mesh_less_sampling_areas.vertices, mesh_less_sampling_areas.faces)[0])
    print(f"Time needed to compute SDF: {time.time() - start}")
    all_points = all_points[unstructured_tolerance < inside_surface_values]


    # sample unstructured regions
    start = time.time()
    print(f"Started computing distance function for {all_points.shape[0]} points")
    inside_surface_values = np.abs(igl.signed_distance(all_points, mesh_wo_less_sampling_areas.vertices, mesh_wo_less_sampling_areas.faces)[0])
    print(f"Time needed to compute SDF: {time.time() - start}")

    return all_points[np.logical_and(structured_tolerance < inside_surface_values, inside_surface_values < max_distance)]

if __name__ == "__main__":
    scene_path = Path("./data/scannet_demo")  # Replace with your actual scene path
    points = get_mask(scene_path)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 8)
    ax.set_zlim(-0.5, 8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Point Cloud')
    plt.show()
