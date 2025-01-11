from typing import TypedDict
from einops import rearrange
from torch import Tensor
from jaxtyping import Float
from abc import ABC, abstractmethod
import torch
from datasets.chunk import image
from datasets import scene
from utils.transformations import invert_pose_batched
from visualization import Visualizer
import visualization
import visualization.images as images
import visualization.mesh as mesh

class GridConfig(TypedDict):
    grid_resolution: float
    grid_size: Float[Tensor, "3"]
    center: Float[Tensor, "3"]


def to_homogeneous(x: Float[Tensor, "... 3"]) -> Float[Tensor, "... 4"]:
    return torch.cat([x, torch.ones_like(x[..., 0:1])], dim=-1)

def apply_rigid_transform(T: Float[Tensor, "4 4"], x: Float[Tensor, "... 4"]) -> Float[Tensor, "... 4"]:    
    return torch.matmul(T, x.unsqueeze(-1)).squeeze(-1)


def project_into_image(K: Float[Tensor, "... 3 3"], T_cw: Float[Tensor, "... 4 4"], x: Float[Tensor, "... 4"]) -> Float[Tensor, "... 2"]:
    T = torch.matmul(K, T_cw[..., :3, :])
    proj = torch.matmul(T, x.unsqueeze(-1)).squeeze(-1)
    return proj

def perspective_division(x: Float[Tensor, "... 3"]) -> Float[Tensor, "... 2"]:
    return x[..., :2] / x[..., 2:3]


class BaseHeuristic(ABC):
    @abstractmethod
    def __call__(self, canonical: list[int], extrinsics_cw: Float[torch.Tensor, "N 4 4"], extrinsics_wc: Float[torch.Tensor, "N 4 4"], intrinsics: Float[torch.Tensor, "N 3 3"], grid_config: GridConfig) -> Float[Tensor, "N 1"]:
        """
        Args:
            canonical: list of indices of currently considered images (indices)
            extrinsics: extrinsics of the images
            intrinsics: intrinsics of the images
        Returns:
            Float[Tensor, "N 1"]: confidence score for the canonical image (gets maximized)
        """
        pass


class AngleHeuristics(BaseHeuristic):
    def __call__(self, current: list[int], extrinsics_cw: Float[torch.Tensor, "N 4 4"], extrinsics_wc: Float[torch.Tensor, "N 4 4"], intrinsics: Float[torch.Tensor, "N 3 3"], grid_config: GridConfig) -> Float[Tensor, "N 1"]:
        # get z axis of all camera
        z = torch.tensor([0, 0, 1]).float().to(extrinsics_wc)
        z = rearrange(z, "F -> 1 F 1")
 
        # get z axis of all cameras
        z_w = torch.matmul(extrinsics_wc[...,:3,:3], z).squeeze(-1)[..., :3]

        # get z axis of all currently considered cameras
        # Tensor will be (len(current), 4)
        z_current = z_w[current]

        # prepare z_w for dot product
        cos_angle = torch.inner(z_w, z_current)

        normed_flipped = (-cos_angle + 1) / 2
        return normed_flipped.mean(dim=-1).unsqueeze(-1)
    
class IsClose(BaseHeuristic):
    def __call__(self, current: list[int], extrinsics_cw: Float[torch.Tensor, "N 4 4"], extrinsics_wc: Float[torch.Tensor, "N 4 4"], intrinsics: Float[torch.Tensor, "N 3 3"], grid_config: GridConfig) -> Float[Tensor, "N 1"]:
        pos_c = extrinsics_wc[..., :3, 3]
        current_pos = pos_c[current]
        
        norm = torch.norm(pos_c.unsqueeze(1) - current_pos, dim=-1)
        
        norm[(norm < 0.4).any(dim=-1)] = -torch.inf
        norm[~(norm < 0.4).any(dim=-1)] = 0
        
        return norm[:, 0].unsqueeze(-1)

class AreaUnderIntrinsics(BaseHeuristic):
    def __call__(self, current: list[int], extrinsics_cw: Float[torch.Tensor, "N 4 4"], extrinsics_wc: Float[torch.Tensor, "N 4 4"], intrinsics: Float[torch.Tensor, "N 3 3"], grid_config: GridConfig) -> Float[Tensor, "N 1"]:
        canonical = current[0]

        # center of the grid
        center_0 = grid_config["center"].float()
        grid_size = grid_config["grid_size"]
        pitch = grid_config["grid_resolution"]
        # get coordinates of the grid vertices in camera frame
        grid_1d = torch.tensor([-1.0, 1.0])
        vertices_n = torch.cartesian_prod(*grid_1d.unsqueeze(0).repeat(3, 1))
        extent = ((pitch * grid_size) / 2).float()
        vertices_0 = center_0 + extent * vertices_n

        # now transform into the world frame of the canonical image
        T_w0 = extrinsics_wc[canonical]
        vertices_w = apply_rigid_transform(T_w0, to_homogeneous(vertices_0))
        center_w = apply_rigid_transform(T_w0, to_homogeneous(center_0))

        # project into image all images
        center_c = project_into_image(intrinsics, extrinsics_cw, center_w)
        vertices_c = project_into_image(intrinsics.unsqueeze(1), extrinsics_cw.unsqueeze(1), vertices_w)

        invalid_mask = (vertices_c[..., 2] < 0).any(dim=-1)

        center_pixel_c = perspective_division(center_c).unsqueeze(1)
        vertices_pixel_c = perspective_division(vertices_c)

        bounds_2d = (apply_rigid_transform(intrinsics, to_homogeneous(torch.tensor([0.0, 0.0])))[..., :2] * 2).unsqueeze(1)
        zero_2d = torch.zeros_like(bounds_2d)

        is_inside = ((center_pixel_c > zero_2d).all(dim=-1) & (center_pixel_c < bounds_2d).all(dim=-1)).squeeze(-1)
        is_inside_vertices = ((vertices_pixel_c > zero_2d).all(dim=-1) & (vertices_pixel_c < bounds_2d).all(dim=-1))
        invalid_mask = invalid_mask | ~is_inside | ~is_inside_vertices.all(dim=-1)

        distance_to_center = torch.norm(vertices_pixel_c - center_pixel_c, dim=-1)
        #distance_to_center[~is_inside_vertices] = -torch.inf
        bounding_radius = torch.max(distance_to_center, dim=-1).values 

        # max_radius = bounds_2d.min(dim=-1).values / 2
        # bounding_radius[invalid_mask] = -torch.inf
        # bounding_radius[~invalid_mask] /= max_radius[~invalid_mask].squeeze(-1)
        
        max_radius = bounding_radius[current[0]]
        bounding_radius[invalid_mask] = -torch.inf
        bounding_radius[~invalid_mask] /= max_radius
        
        bounding_radius[bounding_radius > 1] = 1
        
        return bounding_radius.unsqueeze(-1)
    



Heuristics = {
    "AngleHeuristics": AngleHeuristics,
    "AreaUnderIntrinsics": AreaUnderIntrinsics,
    "IsClose": IsClose
}

if __name__ == "__main__":
    data_config = image.Config.load_from_files([
        "./config/data/base.yaml",
        "./config/data/undistorted_scenes.yaml"
    ])
    
    data_config.seq_len = 8
    base_dataset = scene.Dataset(data_config)
    base_dataset.load_paths() 
    
    area = Heuristics["AreaUnderIntrinsics"]()
    angle = Heuristics["AngleHeuristics"]()
    isClose = Heuristics["IsClose"]()
    
    scene_idx = 0
    data_dict = base_dataset[scene_idx]
    
    grid_config = {
        "grid_resolution": data_config.grid_resolution,
        "grid_size": torch.tensor(data_config.grid_size),
        "center": torch.tensor(data_config.center_point),
    }
    camera_params = data_dict["camera_params"]
    camera_params_list = [camera_params[key] for key in camera_params.keys()]
    extrinsics_cw = torch.Tensor([camera_params[key]["T_cw"] for key in camera_params.keys()])
    extrinsics_wc = invert_pose_batched(extrinsics_cw[:, :3, :3], extrinsics_cw[:, :3, 3])
    intrinsics = torch.Tensor([camera_params[key]["K"] for key in camera_params.keys()])
    #score = heuristic([0], extrinsics_cw, extrinsics_wc, intrinsics, grid_config)
    image_idx_of_interest = extrinsics_cw.__len__() // 2
    winning_idxs = [image_idx_of_interest]
    
    for i in range(data_config.seq_len - 1):
        score_area = area(winning_idxs, extrinsics_cw, extrinsics_wc, intrinsics, grid_config)
        score_angle = angle(winning_idxs, extrinsics_cw, extrinsics_wc, intrinsics, grid_config)
        score_isClose = isClose(winning_idxs, extrinsics_cw, extrinsics_wc, intrinsics, grid_config)
        score = score_area + 0.7*score_angle + score_isClose
        score[winning_idxs] = -torch.inf
        best_idx = torch.argmax(score)
        winning_idxs.append(best_idx.item())

        
    visualizer_config = visualization.Config(log_dir=".visualization", **data_config.model_dump())
    visualizer = Visualizer(visualizer_config)
    
    #visualizer.add_scene(data_dict["scene_name"], opacity=0.1)
    
    for winning_idx in winning_idxs:
        result_dict = {
                            "scene_name": data_dict["scene_name"],
                            "images": [data_dict["path_images"] / list(data_dict["camera_params"].keys())[winning_idx] for winning_idx in winning_idxs],
                            "cameras": [camera_params_list[winning_idx] for winning_idx in winning_idxs],
                            
                        }
        
        # result_dict = {
        #                     "scene_name": data_dict["scene_name"],
        #                     "images": [data_dict["path_images"] / image_name for image_name in list(data_dict["camera_params"].keys())],
        #                     "cameras": camera_params_list,
                            
        #                 }
        
        
        visualizer.add_from_image_dict(result_dict)
        
    visualizer.export_html("out", timestamp=True)
        
            