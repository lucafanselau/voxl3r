import os
from pathlib import Path
import time
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

import pyvista as pv
import trimesh

from extern.scannetpp.common.scene_release import ScannetppScene_Release
from extern.scannetpp.iphone.prepare_iphone_data import extract_depth, extract_masks, extract_rgb
from utils.data_parsing import get_camera_params, get_vertices_labels, load_yaml_munch
from utils.masking import get_mask, get_structures_unstructured_mesh
from utils.transformations import invert_pose, project_image_plane, quaternion_to_rotation_matrix, undistort_image
from utils.visualize import plot_voxel_grid, visualize_mesh, visualize_mesh_without_vertices

class SceneDataset(Dataset):
    def __init__(self, camera="iphone", data_dir="datasets/scannetpp/data", n_points=10000, max_seq_len=20, representation="tdf", threshold_occ=0.0, visualize=False):
        self.camera = camera
        self.data_dir = Path(data_dir)
        self.scenes = [x.name for x in self.data_dir.glob("*") if x.is_dir()]
        self.n_points = n_points
        self.max_seq_len = max_seq_len
        self.cfg = load_yaml_munch(Path(".") / "utils" / "config.yaml")
        self.representation = representation
        
        if threshold_occ > 0.0:
            self.threshold_occ = threshold_occ
        
        self.visualize = visualize

    def __len__(self):
        return len(self.img_labels)

    def get_images_with_3d_point(self, idx, points, image_names = None, tolerance=0.9):           
        c_params = get_camera_params(self.data_dir / self.scenes[idx], self.camera, image_names, self.max_seq_len)

        images_names = []
        pixel_coordinate = []
        camera_params_list = []
        
        uvs_dict, _ = project_image_plane(c_params, points)
        
        for key in c_params.keys():
            c_dict = c_params[key]
                        
            if uvs_dict[key] is not None:
                u, v = uvs_dict[key][:, 0]
                w_min, w_max = (0.5 - tolerance/2) * c_dict['width'], (0.5 + tolerance/2) * c_dict['width']
                h_min, h_max = (0.5 - tolerance/2) * c_dict['height'], (0.5 + tolerance/2) * c_dict['height']
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
        if self.cfg.sdf_strategy == "igl":
            import igl
            inside_surface_values = igl.signed_distance(points, mesh.vertices, mesh.faces)[0]
        elif self.cfg.sdf_strategy == "pysdf":
            from pysdf import SDF
            f = SDF(mesh.vertices, mesh.faces)
            inside_surface_values = f(points)
        print(f"Time needed to compute SDF: {time.time() - start}")

        np.savez(self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz", points=points, sdf=inside_surface_values)
        
    def sample_chunk(self, idx, image_name, center = np.array([0.0,0.0,1.25]), size = np.array([1.5,1.0,2.0]), visualize=False):
 
        if not (self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz").exists():
            self.sample_scene(idx)

        data = np.load(self.data_dir / self.scenes[idx] / "scans" / "sdf_pointcloud.npz") 
        points, sdf = data['points'], data['sdf']
        
        c_dict = get_camera_params(self.data_dir / self.scenes[idx], self.camera, image_name, 1)[image_name]
        transformation = c_dict['T_cw']
        _, _, back_transformation = invert_pose(c_dict['R_cw'], c_dict['t_cw'])
        
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points = (transformation @ points.T).T[:, :3]
        
        sdf = sdf[(np.abs(points - center) < size/2).all(axis=1)]
        df = np.abs(sdf)
        
        gt = None
        if self.representation == "tdf":
            df[df > 1] = 1
            gt = np.abs(df)
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
        
            for i, offset in enumerate(size):
                plane_origin, plane_normal = np.zeros(3), np.zeros(3)
                plane_origin[i], plane_normal[i] = offset/2, 1
                mesh = mesh.slice_plane(center-plane_origin, plane_normal)
                mesh = mesh.slice_plane(center+plane_origin, -plane_normal)
            
            mesh.apply_transform(back_transformation)
        
        vec_center =  np.array([*center.flatten(),  1])
        P_center = (back_transformation @ vec_center).flatten()[:3]
        image_names, pixel_coordinate, camera_params_list = self.get_images_with_3d_point(idx, P_center, image_names=image_name, tolerance=0.8)
        
        image_file = "rgb" if self.camera == "iphone" else "images"
        image_names = [self.data_dir / self.scenes[idx] / self.camera / image_file / image_name for image_name in image_names]
        
        return (points, gt), (image_names, camera_params_list, P_center), mesh  
        
    def get_index_from_scene(self, scene_name):
        return self.scenes.index(scene_name)

    def __getitem__(self, idx):
        path_camera = self.data_dir / self.scenes[idx]  / self.camera
        
        # check if the data has already been extracted
        if self.camera == "iphone" and not ((path_camera / "rgb").exists() and (path_camera / "rgb_masks").exists() and (path_camera / "depth").exists()):
                self.extract_iphone(idx)
    
        # sample a random chunk of the scene and return points, sdf values, and images with camera parameters
        image_folder = "resized_images" if self.camera == "dslr" else "rgb"
        image_names = sorted(os.listdir(path_camera / image_folder))
        image_len = len(image_names) if self.camera == "dslr" else len(image_names) // 10
        # generate random number using image_len
        np.random.seed(0)
        image_name = image_names[np.random.randint(0, image_len) if self.camera == "dslr" else np.random.randint(0, image_len) * 10]
        
        training, images, mesh = self.sample_chunk(idx, image_name, visualize=self.visualize)
        
        return {
            'mesh': mesh,
            'training_data': training,
            'images': images,
        }

def get_image_to_random_vertice(mesh_path):
    mesh = pv.read(mesh_path)
    np.random.seed(42)
    vertices = mesh.points
    random_indices = np.random.randint(0, vertices.shape[0])
    return vertices[random_indices]

def plot_random_training_example(dataset, idx):    
    data_dict = dataset[idx]
    mesh = data_dict['mesh']
    points, gt = data_dict['training_data']
    image_names, camera_params_list, P_center = data_dict['images']
    
    visualize_mesh(pv.wrap(mesh), images=image_names, camera_params_list=camera_params_list, heat_values=gt, point_coords=points)
    
def plot_mask(dataset, idx):
    points = get_mask(dataset.data_dir / dataset.scenes[idx])
    visualize_mesh(dataset.data_dir / dataset.scenes[idx] / "scans" / "mesh_aligned_0.05.ply", point_coords=points)
    
def plot_occupency_grid(dataset, idx):
    assert dataset.representation == "occ", "The representation must be used with representation='occ'"
    data_dict = dataset[idx]
    points, gt = data_dict['training_data']    
    plot_voxel_grid(points, gt, resolution=0.01, ref_mesh=data_dict["mesh"])
    
if __name__ == "__main__":
    dataset = SceneDataset(camera="iphone", n_points=300000, threshold_occ=0.01, representation="occ", visualize=True)
    
    idx = dataset.get_index_from_scene("8b2c0938d6")
    # plot_mask(dataset, idx)
    # plot_random_training_example(dataset, idx)
    plot_occupency_grid(dataset, idx)