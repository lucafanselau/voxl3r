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
    mast3r_stat_file: Optional[str] = None
    mast3r_grid_resolution: float = 0.04
    max_points_in_voxel: int = 16


class OutputDict(TypedDict):
    X: Tensor
    Y: Tensor
    
def point_transform_collate_fn(batch: list) -> OutputDict:
    # return pts_0, pts_conf, pts_desc, pts_desc_conf, pts_in_grid, voxel_ids_final
    occupied_voxel_ids = torch.Tensor([ele[-1].shape[0] for ele in batch]).long()
    voxel_grid_start_indices = occupied_voxel_ids.cumsum(0) - occupied_voxel_ids
    
    point_count = torch.Tensor([ele[0].shape[0] for ele in batch]).long()
    point_start_indices = point_count.cumsum(0) - point_count
    
    voxel_ids_final = torch.cat([ele[-1] for ele in batch])
    pts_in_grid = torch.cat([ele[-2] for ele in batch])
    
    empty_entries = (pts_in_grid == -1)
    pts_in_grid = pts_in_grid + point_start_indices.repeat_interleave(occupied_voxel_ids).unsqueeze(-1)
    pts_in_grid[empty_entries] = -1
        
    features = torch.cat(
        [
            torch.cat([ele[0] for ele in batch]),
            torch.cat([ele[1] for ele in batch]),
            torch.cat([ele[2] for ele in batch]),
            torch.cat([ele[3] for ele in batch])
        ], dim=-1
    )
    
    return features, pts_in_grid, voxel_grid_start_indices

class PointBasedTransform(nn.Module):
    def __init__(self, config: PointBasedTransformConfig):
        super().__init__()
        self.config = config
        
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
        T_0w = T_cw[0] # all points are transformer into this camera frame
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
        
        num_voxels = mast3r_grid_size[2]*mast3r_grid_size[1]*mast3r_grid_size[0]
        # calculate unique hash for each voxel
        # p_x * (grid_size[2]*grid_size[1]) + p_y * grid_size[2] + p_z
        # reverse: voxel_id // (grid_size[2]*grid_size[1]), (voxel_id % (grid_size[2]*grid_size[1])) // grid_size[2], (voxel_id % (grid_size[2]*grid_size[1])) % grid_size[2]
        voxel_ids = pts_0_idx[:, 0]*(mast3r_grid_size[2]*mast3r_grid_size[1]) + pts_0_idx[:, 1] * mast3r_grid_size[2] + pts_0_idx[:, 2]
        
        # Sort by confidence (dont know if it has to be stable here)
        pts_conf, idx_conf = pts_conf.sort(dim=0, descending=True, stable=True)
        voxel_ids = voxel_ids[idx_conf].squeeze(-1)
        point_ids = torch.arange(voxel_ids.shape[0])[idx_conf].squeeze(-1)
        
        # Sort by voxel ID so identical voxel IDs are consecutive
        # we get for each point the voxel id (voxel_id_sorted)
        voxel_id_sorted, sort_idx = voxel_ids.sort(dim=0, descending=False, stable=True)
        point_id_sorted = point_ids[sort_idx]
        
        # Count how many point indices go to each voxel
        counts = voxel_id_sorted.bincount(minlength=num_voxels)
        
        # Compute a starting offset for each voxel using a prefix sum
        # offsets[i] = the first index in point_id_sorted for voxel i 
        # (where do to points for voxel i start) (max. is num_points)
        offsets = counts.cumsum(0) - counts  # shape [num_voxels]
        
        # Offset goes from 0 to num_voxel, since we are only interested in the voxels
        # that have points in them we use voxel_id_sorted
        position_in_voxel = torch.arange(len(voxel_id_sorted), device=voxel_id_sorted.device)
        # gives back an array with the row position of the points so for example:
        # [0, 1, 0, 1, 2, 3, 0, 0, 1, 2, 3, 4, 5]
        # in this example the first two points in point_id_sorted belong to the first voxel
        # the next four points belong to the second voxel and so on
        position_in_voxel = position_in_voxel - offsets[voxel_id_sorted]
            
        max_points_in_voxel = self.config.max_points_in_voxel 
        
        if max_points_in_voxel == -1:
            max_points_in_voxel = counts.max().item()
            

        # position_in_voxel is also sorted on confidence so the points with the 
        # lowest confidence are removed first
        valid_mask = (position_in_voxel < max_points_in_voxel)
        voxel_ids_final = voxel_id_sorted[valid_mask]
        point_ids_final = point_id_sorted[valid_mask]
        position_final = position_in_voxel[valid_mask]
        
        # reorder the points
        pts_0 = pts_0[point_ids_final]
        pts_conf = pts_conf[point_ids_final]
        pts_desc = pts_desc[point_ids_final]
        pts_desc_conf = pts_desc_conf[point_ids_final]
    
        _, voxel_ids_final_no_offsets = voxel_ids_final.unique_consecutive(return_inverse=True)
        
        if len(voxel_ids_final_no_offsets) != 0:
            num_occupied_voxels = voxel_ids_final_no_offsets[-1] + 1
            
            pts_in_grid = torch.full(
                (num_occupied_voxels, max_points_in_voxel),
                -1,
                dtype=torch.long,
                device=voxel_id_sorted.device
            )
            
            # gives us for each occupied voxel the point ids that belong to it
            # this way all features can be easily accessed
            pts_in_grid[voxel_ids_final_no_offsets, position_final] = torch.arange(point_ids_final.shape[0], device=voxel_ids_final.device)
            
            # each voxel in pts_in_grid ha a unique identifier represented in this list
            voxel_ids_final = torch.unique(voxel_ids_final, sorted=False)
        else:
            pts_in_grid = torch.Tensor([])
            voxel_ids_final = torch.Tensor([]).long()

        visualize = False
        
        if not visualize:
            return pts_0, pts_conf, pts_desc, pts_desc_conf, pts_in_grid, voxel_ids_final
        
        else:
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
            
            # voxel id to x,y,z indices in grid
            voxel_id = voxel_ids_final[most_occupied_voxel]
            x, y, z = voxel_id // (mast3r_grid_size[2]*mast3r_grid_size[1]), (voxel_id % (mast3r_grid_size[2]*mast3r_grid_size[1])) // mast3r_grid_size[2], (voxel_id % (mast3r_grid_size[2]*mast3r_grid_size[1])) % mast3r_grid_size[2]
            coordinate = torch.tensor([x, y, z], device=voxel_id.device)
            voxel_center_pt = coordinate * self.config.mast3r_grid_resolution + self.config.mast3r_grid_resolution / 2 + center - (cube_size / 2)
            voxel_center = pv.PolyData(voxel_center_pt.detach().cpu().numpy())
            
            
            voxel_point_cloud = pv.PolyData(pts_0[pts_in_grid[most_occupied_voxel][pts_in_grid[most_occupied_voxel] != -1]].detach().cpu().numpy())
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
            data["occupancy_grid"] = torch.zeros_like(data["occupancy_grid"])
            data["occupancy_grid"][0, x//2, y//2, z//2] = 1
            visualizer.add_from_occupancy_dict(data, opacity=0.5)
            
            
            # Save as HTML
            plotter.export_html(f'./.visualization/point_cloud_visualization_{max_points_in_voxel}_{data["scene_name"]}_{data["file_name"]}.html')
            
            print("Visualization saved")

        
