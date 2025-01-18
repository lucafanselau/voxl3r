from pathlib import Path
from einops import rearrange
import torch
from torch import nn, Tensor
from typing import Optional, Union, TypedDict
from datasets.chunk import image, mast3r, occupancy_revised as occupancy
from datasets.transforms import images
from utils.chunking import compute_coordinates
from utils.config import BaseConfig
from utils.transformations import extract_rot_trans_batched, invert_pose_batched


class PointBasedTransformConfig(mast3r.Config, image.Config):
    # scales the confidences effect on the feature mask
    alpha: float = 1.0
    mast3r_stat_file: Optional[str] = None


class OutputDict(TypedDict):
    X: Tensor
    Y: Tensor

class PointBasedTransform(nn.Module):
    def __init__(self, config: PointBasedTransformConfig):
        super().__init__()
        self.config = config
        self.transformation_transform = images.StackTransformations()
        
        if self.config.mast3r_stat_file is not None:
            self.mast3r_stats = torch.load(self.config.mast3r_stat_file)


    def __call__(self, data: dict) -> OutputDict: 
        grid_size = torch.tensor(data["grid_size"])
        center = data["center"].clone().detach()
        pitch = data["resolution"]

        # load images from pairwise_predictions and associated transformations
        res_dict_1 = data["pairwise_predictions"][0]
        res_dict_2 = data["pairwise_predictions"][1]
        
        if self.mast3r_stats is not None:
            res_dict_1["desc"] = (res_dict_1["desc"] - self.mast3r_stats["desc1_mean"]) / self.mast3r_stats["desc1_std"]
            res_dict_2["desc"] = (res_dict_2["desc"] - self.mast3r_stats["desc2_mean"]) / self.mast3r_stats["desc2_std"]
            res_dict_1["desc_conf"] = (res_dict_1["desc_conf"] - self.mast3r_stats["desc_conf1_mean"]) / self.mast3r_stats["desc_conf1_std"]
            res_dict_2["desc_conf"] = (res_dict_2["desc_conf"] - self.mast3r_stats["desc_conf2_mean"]) / self.mast3r_stats["desc_conf2_std"]
        
        image_names = [Path(name).name for name in data["images"]]
        T_cw = torch.cat([torch.tensor(data["cameras"][i]["T_cw"]).unsqueeze(0) for i in range(len(data["cameras"]))])
        T_0w = T_cw[0] # only first centered is supported yet
        pairs = data["pairs_image_names"]

        if self.config.pair_matching != "first_centered":
            T_pair_cw = torch.stack([T_cw[image_names.index(pair[0])] for pair in pairs])
            T_0c = torch.matmul(T_0w, invert_pose_batched(*extract_rot_trans_batched(T_pair_cw)))
    
        pts_c = torch.cat([res_dict_1[f"pts3d"], res_dict_2[f"pts3d"]])
        pts_conf = torch.cat([res_dict_1[f"conf"], res_dict_2[f"conf"]])
        pts_desc = torch.cat([res_dict_1[f"desc"], res_dict_2[f"desc"]])
        pts_desc_conf = torch.cat([res_dict_1[f"desc_conf"], res_dict_2[f"desc_conf"]])

        # transform all points into the same space with T_0c
        if self.config.pair_matching != "first_centered":
            to_homogeneous = lambda x: torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            points_homogeneous = to_homogeneous(pts_c).unsqueeze(-1)

            pts_0 = torch.matmul(rearrange(T_0c, "NP F D -> NP 1 1 1 F D").float(), points_homogeneous).squeeze(-1)[..., :3]
        else:
            pts_0 = pts_c
            
        cube_size = (torch.tensor(grid_size, dtype=pts_0.dtype, device=pts_0.device) * pitch) / 2.0
        min_corner = center - cube_size
        max_corner = center + cube_size
        
        pts_0 = pts_0.reshape(-1, 3)
        
        mask = ((pts_0[:, 0] >= min_corner[0]) & (pts_0[:, 0] <= max_corner[0]) &
                (pts_0[:, 1] >= min_corner[1]) & (pts_0[:, 1] <= max_corner[1]) &
                (pts_0[:, 2] >= min_corner[2]) & (pts_0[:, 2] <= max_corner[2]))
        
        pts_0 = pts_0[mask]
        
        visualize = True
        if visualize:
            # take points_0 as a point cloud and visualize it and save it as a html file
            import pyvista as pv
            
            # Create a plotter
            plotter = pv.Plotter(notebook=True)
            
            # Convert points to numpy and reshape
            points_np = pts_0.detach().cpu().numpy()
            
            # Create point cloud
            point_cloud = pv.PolyData(points_np)
            
            # Add points to plotter
            plotter.add_points(point_cloud, point_size=5)
            
            # Set up camera and lighting
            plotter.camera_position = 'xy'
            plotter.enable_eye_dome_lighting()
            
            # Save as HTML
            plotter.export_html('./.visualization/point_cloud_visualization.html')

        
        # coordinates of voxel grid in the frame of the first image
        # coordinates is (3, x, y, z)
        coordinates = torch.from_numpy(compute_coordinates(
            grid_size.numpy(),
            center.numpy(),
            pitch,
            grid_size[0].item(),
            to_world_coordinates=None
        )).float().to(grid_size.device)

        center_0 = torch.tensor(data["center"]).float()
        grid_size = torch.tensor(data["grid_size"]).float()
        pitch = torch.tensor(data["resolution"]).float()
        # get coordinates of the grid vertices in camera frame
        grid_1d = torch.tensor([-1.0, 1.0])
        extent = ((pitch * grid_size) / 2).float()
        lower_bound_0 = center_0 - extent
        upper_bound_0 = center_0 + extent

        alpha = 1.0

        i = 0
        P, H, W, _3 = points_0[i].shape
        # mask points with confidence
        p = points_0[i]
        p = p[conf_norm[i] > 2.5]
        confs = conf_norm[i][conf_norm[i] > 2.5]
        confs = rearrange(confs, "N -> N 1 1 1 1")
        p = rearrange(p, "N C -> N 1 1 1 C")
        c = rearrange(coordinates, "C X Y Z -> 1 X Y Z C")

        valid_points = ((points_0 > lower_bound_0) & (points_0 < upper_bound_0)).all(dim=-1)

        dist = torch.distributions.Normal(0, 1)

        log_probs = dist.log_prob((c - p) / (alpha * confs * pitch))


        for i in range(points_0.shape[0]):
            pass            


        
        # get T_0w from data
        # this reads as from the images get the transformations, then the one for the first (0) image and of this the full transformation matrix
        # T_0w = torch.tensor(data["images"][1][0]["T_cw"])


        # H, W = images.shape[-2:]
        # transformations, T_cw, K = self.transformation_transform(image_dict, new_shape=torch.Tensor((H, W)))

        breakpoint()
        