from pathlib import Path
from einops import rearrange
import torch
from torch import nn, Tensor
from typing import Optional, Union, TypedDict
from datasets import scene
from datasets.chunk import image, mast3r, occupancy_revised as occupancy
from datasets.transforms import images
from datasets.transforms.smear_images import SmearMast3rConfig
from utils.chunking import compute_coordinates
from utils.config import BaseConfig
from utils.transformations import extract_rot_trans_batched, invert_pose, invert_pose_batched


class PointBasedTransformConfig(mast3r.Config, image.Config, scene.Config, SmearMast3rConfig):
    # scales the confidences effect on the feature mask
    alpha: float = 1.0
    mast3r_stat_file: Optional[str] = None
    mast3r_grid_resolution: float = 0.02
    max_points_in_voxel: int = 32


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
        
        # check if this is not a integer
        if ((grid_size / (self.config.mast3r_grid_resolution / pitch)) % 1).any():
            raise ValueError("Mast3r grid size must be a integer")
        
        mast3r_grid_size = (grid_size / (self.config.mast3r_grid_resolution / pitch)).int()

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

        # transform all points into the same space with T_0c
        if self.config.pair_matching != "first_centered":
            to_homogeneous = lambda x: torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            points_homogeneous = to_homogeneous(pts_c).unsqueeze(-1)

            pts_0 = torch.matmul(rearrange(T_0c, "NP F D -> NP 1 1 1 F D").float(), points_homogeneous).squeeze(-1)[..., :3]
        else:
            pts_0 = pts_c
         
        pts_0 =  pts_0.reshape(-1, 3)   
        cube_size = (torch.tensor(grid_size, dtype=pts_0.dtype, device=pts_0.device) * pitch)
        min_corner = center - (cube_size / 2)
        max_corner = center + (cube_size / 2)
        
        pts_0_idx = ((pts_0 - min_corner) / self.config.mast3r_grid_resolution).int()
        
        mask = ((pts_0_idx[:, 0] >= 0) & (pts_0_idx[:, 0] < mast3r_grid_size[0]) &
                (pts_0_idx[:, 1] >= 0) & (pts_0_idx[:, 1] < mast3r_grid_size[1]) &
                (pts_0_idx[:, 2] >= 0) & (pts_0_idx[:, 2] < mast3r_grid_size[2]))
        
        
        pts_0 = pts_0[mask]
        pts_0_idx = pts_0_idx[mask]
        pts_conf = torch.cat([res_dict_1[f"conf"], res_dict_2[f"conf"]]).reshape(-1, 1)[mask]
        pts_desc = torch.cat([res_dict_1[f"desc"], res_dict_2[f"desc"]]).reshape(-1, 24)[mask]
        pts_desc_conf = torch.cat([res_dict_1[f"desc_conf"], res_dict_2[f"desc_conf"]]).reshape(-1, 1)[mask]
        
        coords = torch.cartesian_prod(
            torch.arange(mast3r_grid_size[0]),
            torch.arange(mast3r_grid_size[1]),
            torch.arange(mast3r_grid_size[2])
        )
        
        num_voxels = coords.shape[0]
        # pts_sequence_mask = (pts_0_idx.unsqueeze(1) == coords.unsqueeze(0)).all(dim=-1)
        voxel_ids = pts_0_idx[:, 0]*(mast3r_grid_size[2]*mast3r_grid_size[1]) + pts_0_idx[:, 1] * mast3r_grid_size[2] + pts_0_idx[:, 2]
        
        # Sort by confidence
        pts_conf, idx_conf = pts_conf.sort(dim=0, descending=True, stable=True)
        voxel_ids = voxel_ids[idx_conf].squeeze(-1)
        point_ids = torch.arange(voxel_ids.shape[0])[idx_conf].squeeze(-1)
        
        # Sort by voxel ID so identical voxel IDs are consecutive
        # we get for each point the voxel id (voxel_id_sorted)
        voxel_id_sorted, sort_idx = voxel_ids.sort(dim=0, descending=False, stable=True)
        point_id_sorted = point_ids[sort_idx].squeeze(-1)
        
        # Count how many point indices go to each voxel
        counts = voxel_id_sorted.bincount(minlength=num_voxels)
        
        # Compute a starting offset for each voxel using a prefix sum
        # offsets[i] = the first column index in pts_in_grid for voxel i
        offsets = counts.cumsum(0) - counts  # shape [num_voxels]
        
        position_in_voxel = torch.arange(len(voxel_id_sorted), device=voxel_id_sorted.device)
        position_in_voxel = position_in_voxel - offsets[voxel_id_sorted]
            
        max_points_in_voxel = self.config.max_points_in_voxel 
        if max_points_in_voxel == -1:
            max_points_in_voxel = counts.max().item()
            
        pts_in_grid = torch.full(
            (num_voxels, max_points_in_voxel),
            -1,
            dtype=torch.long,
            device=voxel_id_sorted.device
        )

        valid_mask = (position_in_voxel < max_points_in_voxel)
        voxel_ids_final = voxel_id_sorted[valid_mask]
        point_ids_final = point_id_sorted[valid_mask]
        position_final = position_in_voxel[valid_mask]
        
        # reorder the points
        pts_0 = pts_0[point_ids_final]
        pts_conf = pts_conf[point_ids_final]
        pts_desc = pts_desc[point_ids_final]
        pts_desc_conf = pts_desc_conf[point_ids_final]
    
        pts_in_grid[voxel_ids_final, position_final] = torch.arange(point_ids_final.shape[0], device=voxel_ids_final.device)
        
        #pts_feature = torch.cat([pts_0, pts_conf.unsqueeze(-1), pts_desc, pts_desc_conf.unsqueeze(-1)], dim=-1)
        
        visualize = True
        if visualize:
            # take points_0 as a point cloud and visualize it and save it as a html file
            import pyvista as pv
            
            _, _, T_w0 = invert_pose(T_0w[:3, :3], T_0w[:3, 3])
            # Create a plotter
            plotter = pv.Plotter(notebook=True)
            
            # Create point cloud
            point_cloud = pv.PolyData(pts_0.detach().cpu().numpy())
            
            # Add points to plotter
            point_cloud.transform(T_w0)
            if False:
                point_cloud["Heat"] = pts_conf.detach().cpu().numpy()
                plotter.add_mesh(
                    point_cloud,
                    scalars="Heat",
                    cmap="viridis",
                    point_size=8,
                    render_points_as_spheres=True,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "Heat", "shadow": True},
                )
            else:
                plotter.add_points(point_cloud, point_size=3)
            
            most_occupied_voxel = (pts_in_grid != torch.Tensor([-1])).sum(dim=-1).argmax()
            
            voxel_center_pt = coords[most_occupied_voxel] * self.config.mast3r_grid_resolution + center - (cube_size / 2)
            voxel_center = pv.PolyData(voxel_center_pt.detach().cpu().numpy())
            
            voxel_point_cloud = pv.PolyData(pts_0[pts_in_grid[most_occupied_voxel]].detach().cpu().numpy())
            voxel_point_cloud.transform(T_w0)
            voxel_center.transform(T_w0)
            
            plotter.add_points(voxel_point_cloud, point_size=8, color='red')
            plotter.add_points(voxel_center, point_size=10, color='green')
            
            # Set up camera and lighting
            plotter.camera_position = 'xy'
            plotter.enable_eye_dome_lighting()
            
            from visualization import mesh, occ
            #visualizer = mesh.Visualizer(mesh.Config(log_dir=".visualization", **self.config.model_dump()))
            #visualizer.plotter = plotter
            #visualizer.add_scene(data["scene_name"])
            visualizer = occ.Visualizer(occ.Config(log_dir=".visualization", **self.config.model_dump()))
            visualizer.plotter = plotter
            visualizer.add_from_occupancy_dict(data, opacity=0.5)
            
            
            # Save as HTML
            plotter.export_html(f'./.visualization/point_cloud_visualization_{max_points_in_voxel}_{data["scene_name"]}_{data["file_name"]}.html')

        