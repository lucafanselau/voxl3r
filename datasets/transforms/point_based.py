from pathlib import Path
from einops import rearrange
import torch
from torch import nn, Tensor
from typing import Union, TypedDict
from datasets.chunk import image, mast3r, occupancy_revised as occupancy
from datasets.transforms import images
from utils.chunking import compute_coordinates
from utils.config import BaseConfig
from utils.transformations import extract_rot_trans_batched, invert_pose_batched


class PointBasedTransformConfig(BaseConfig):
    # scales the confidences effect on the feature mask
    alpha: float = 1.0


class OutputDict(TypedDict):
    X: Tensor
    Y: Tensor

class PointBasedTransform(nn.Module):
    def __init__(self, config: PointBasedTransformConfig):
        super().__init__()
        self.config = config
        self.transformation_transform = images.StackTransformations()

    def __call__(self, data: dict) -> OutputDict: 
        grid_size = torch.tensor(data["grid_size"])
        center = data["center"].clone().detach()
        pitch = data["resolution"]

        # load images from pairwise_predictions and associated transformations
        res_dict = {
            **data["pairwise_predictions"][0],
            **data["pairwise_predictions"][1],
        }
        pred = data["pairwise_predictions"]

        # transform a single pair into a 3d voxel grid
        # 

        # first get all the 3d points and associated transformations
        keys = list(res_dict.keys())

        T_0w = torch.tensor(data["cameras"][0]["T_cw"])

        # TODO: this has to be replaced with actual pair selection
        image_names = [Path(name).name for name in data["images"]]
        T_cw = torch.cat([torch.tensor(data["cameras"][i]["T_cw"]).unsqueeze(0) for i in range(len(data["cameras"]))])
        pairs = [(image_names[i], image_names[i+1]) for i in range(0, len(image_names), 2)]

        # transformations from each seminal image (eg. the first image in earch pair) and the 0 image
        # if all the first images are the same, then we can use the same transformation for all pairs (eg eye(4, 4))
        if len(set(image_names[0::2])) == 1:
            T_0c = torch.eye(4, 4).unsqueeze(0).repeat(len(data["images"][1]) // 2, 1, 1)
        else:
            T_pair_cw = T_cw[0::2]
            # generally this should be T_0w @ invert_pose(T_pair_cw)
            T_0c = torch.matmul(T_0w, invert_pose_batched(*extract_rot_trans_batched(T_pair_cw)))

        # flat tensor of all features and points
        points = torch.stack([torch.stack([
                    res_dict[f"pts3d_{pair[0]}"],
                    res_dict[f"pts3d_{pair[1]}"],
                ]) for pair in pairs])

        # confidence is (1 + exp(C_n)) where C_n is the normalized confidence [0, 1]
        conf = torch.stack([torch.stack([
                    res_dict[f"conf_{pair[0]}"],
                    res_dict[f"conf_{pair[1]}"],
                ]) for pair in pairs])
        
        conf_norm = torch.log(conf - 1)
        
        features = torch.stack([torch.stack([
                    res_dict[f"desc_{pair[0]}"],
                    res_dict[f"desc_{pair[1]}"],
                ]) for pair in pairs])

        # transform all points into the same space with T_0c
        to_homogeneous = lambda x: torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        points_homogeneous = to_homogeneous(points).unsqueeze(-1)

        points_0 = torch.matmul(rearrange(T_0c, "NP F D -> NP 1 1 1 F D").float(), points_homogeneous).squeeze(-1)[..., :3]

        visualize = False
        if visualize:
            # take points_0 as a point cloud and visualize it and save it as a html file
            import pyvista as pv
            
            # Create a plotter
            plotter = pv.Plotter(notebook=True)
            
            # Convert points to numpy and reshape
            points_np = points_0.detach().cpu().numpy()
            points_np = points_np.reshape(-1, 3)  # Flatten to (N, 3)
            
            # Create point cloud
            point_cloud = pv.PolyData(points_np)
            
            # Add points to plotter
            plotter.add_points(point_cloud, point_size=5)
            
            # Set up camera and lighting
            plotter.camera_position = 'xy'
            plotter.enable_eye_dome_lighting()
            
            # Save as HTML
            plotter.export_html('point_cloud_visualization.html')

        
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
        