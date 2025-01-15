from typing import Optional, Tuple
from beartype import beartype
from jaxtyping import jaxtyped, Bool, Float, Int
import pyvista as pv
import numpy as np
from torch import Tensor
import trimesh

from utils.chunking import compute_coordinates
from . import base
from einops import rearrange
from utils.transformations import from_rot_trans, invert_pose, extract_rot_trans
import torch

class Config(base.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)
        
    def add_from_voxel_grid(self, voxel_grid, opacity: Optional[float] = 0.5, to_world: Optional[bool] = True, transformed=False) -> None:
        mesh = voxel_grid.apply_transform(voxel_grid.transform).as_boxes()
        pv_mesh = pv.wrap(mesh)
        self.plotter.add_mesh(pv_mesh, opacity=opacity)

    def add_from_occupancy_dict(self, occupancy_dict: dict, opacity: Optional[float] = 0.5, to_world: Optional[bool] = True, transformed=False) -> None:

        if transformed:
            data = occupancy_dict["verbose"]["data_dict"]
            occupancy = data["occupancy_grid"]
        else:
            data = occupancy_dict
            occupancy = data["occupancy_grid"]
            
        center = data["center"]
        # T_center = from_rot_trans(torch.eye(3, 3), center)

        T_cw = data["cameras"][0]["T_cw"]
        _, _, T_wc = invert_pose(*extract_rot_trans(T_cw))

        # T_world_object = T_wc @ T_center 

        # occupancy = rearrange(occupancy, "1 X Y Z -> 1 Y X Z")

        self.add_occupancy(occupancy.int(), torch.Tensor(T_wc) if to_world else None, pitch=data["resolution"], origin=torch.Tensor(center), opacity=opacity)
        
    def add_from_occupancy_dict_as_points(self, occupancy_dict: dict, opacity: Optional[float] = 0.5, to_world: Optional[bool] = True, p_size=15, color: Optional[str] = "red", with_transform: Optional[bool] = True) -> None:
    
        if with_transform:
            data_dict = occupancy_dict["verbose"]["data_dict"]
            coordinates = occupancy_dict["verbose"]["coordinates"]
            occupancy_grid= rearrange(data_dict["occupancy_grid"], "1 X Y Z -> (X Y Z) 1")
        else:
            T_cw = occupancy_dict["images"][1][0]["T_cw"]
            _, _, T_wc = invert_pose(*extract_rot_trans(T_cw))
            coordinates = compute_coordinates(
                np.array(occupancy_dict["grid_size"]),
                np.array(occupancy_dict["center"]),
                np.array(occupancy_dict["resolution"]),
                occupancy_dict["grid_size"][0],
                to_world_coordinates=T_wc if to_world else None,
            )
            occupancy_grid= rearrange(occupancy_dict["occupancy_grid"], "1 X Y Z -> (X Y Z) 1")
        
        coordinates = rearrange(coordinates, "C X Y Z -> (X Y Z) C")
        point_coords = np.asarray(coordinates[occupancy_grid.bool().flatten()])
        points = pv.PolyData(point_coords)
        self.plotter.add_mesh(
                points, color=color, point_size=p_size, render_points_as_spheres=True, opacity=opacity
            )
        

    def add_points(self, points: Float[Tensor, "N 3"], color: Optional[str] = "red", p_size: Optional[float] = 15, opacity: Optional[float] = 1.0) -> None:
        points = pv.PolyData(points.numpy())
        self.plotter.add_mesh(points, color=color, point_size=p_size, render_points_as_spheres=True, opacity=opacity)

    def _create_voxel_grid(
        self, values: np.ndarray, origin: np.ndarray = np.array([0.0, 0.0, 0.0]), pitch: float = 1.0
    ) -> pv.StructuredGrid:
        """Create a PyVista structured grid from voxel values.

        The grid is created with one more point than cells in each dimension,
        since points define the corners of cells.
        """
        X_shape, Y_shape, Z_shape = values.shape[-3:]

        # offset origin by - 0.5 * pitch * size
        origin = origin - 0.5 * pitch * np.array([X_shape, Y_shape, Z_shape])

        # Create coordinate arrays with one more point than cells in each dimension
        x = np.arange(X_shape + 1) * pitch + origin[0]
        y = np.arange(Y_shape + 1) * pitch + origin[1]
        z = np.arange(Z_shape + 1) * pitch + origin[2]

        # Create mesh grid of points
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)

        return grid
    
    def _add_outline(self, outline, color: str = "black", line_width: float = 1.0):
        """Add an outline box around the grid."""
        self.plotter.add_mesh(
            outline,
            color=color,
            style="wireframe",
            line_width=line_width,
            render_lines_as_tubes=True,
        )

    def visualize_batch(
        self,
        voxels: Float[Tensor, "batch channels depth height width"],
        mask: Optional[Float[Tensor, "batch depth height width"]] = None,
        origin: Float[Tensor, "batch 3"] = torch.zeros(3),
        T_world_object: Optional[Float[Tensor, "batch 4 4"]] = None,
        pitch: float = 1.0,
        opacity: Optional[float] = 0.5,
    ) -> None:
        """Visualize a batch of voxel grids"""
        voxels = voxels.cpu().numpy()
        batch_size = voxels.shape[0]

        for i in range(batch_size):
            # Calculate offset for this grid
            offset = origin[i].cpu().numpy()

            # Create and add grid
            current_mask = mask[i] if mask is not None else None
            grid = self._create_voxel_grid(voxels[i], origin=offset, pitch=pitch)
            outline = grid.outline()

            if voxels.shape[1] == 3:  # RGB
                rgb = np.moveaxis(voxels[i], 0, -1)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                grid.cell_data["RGB"] = rgb.reshape(-1, 3, order="F")

                if current_mask is not None:
                    current_mask = current_mask.cpu().numpy()
                    grid.cell_data["mask"] = current_mask.flatten(order="F")
                    grid = grid.threshold(0.5, scalars="mask")

                if T_world_object is not None:
                    grid.transform(T_world_object[i].cpu().numpy(), inplace=True)

                if (grid.n_points > 0):
                    self.plotter.add_mesh(
                        grid,
                        rgb=True,
                        opacity=opacity,
                    )
            else:
                grid.cell_data["values"] = voxels[i, 0].flatten(order="F")

                if current_mask is not None:
                    current_mask = current_mask.cpu().numpy()
                    grid.cell_data["mask"] = current_mask.flatten(order="F")
                    grid = grid.threshold(0.5, scalars="mask")

                if T_world_object is not None:
                    grid.transform(T_world_object[i].cpu().numpy(), inplace=True)

                if (grid.n_points > 0):
                    self.plotter.add_mesh(
                        grid,
                        scalars="values",
                        opacity=opacity,
                    )

            # Add outline box
            if T_world_object is not None:
                outline.transform(T_world_object[i].cpu().numpy(), inplace=True)
            self._add_outline(outline) 

    def add_from_scene_occ(self, dict):
        grid = dict["voxel_grid"]
        self.plotter.add_mesh(pv.wrap(grid.as_boxes()), opacity=0.5)
        return
        occupancy = torch.tensor(grid.matrix).unsqueeze(0)
        transform = torch.tensor(grid.transform).unsqueeze(0)
        self.visualize_batch(torch.ones_like(occupancy).unsqueeze(0), mask=occupancy, pitch=grid.pitch[0].item(), opacity=0.5)

    @jaxtyped(typechecker=beartype)
    def add_occupancy(self, occupancy: Int[Tensor, "1 X Y Z"], T_world_object: Optional[base.Transformation] = None, origin: Float[Tensor, "3"] = torch.zeros(3), pitch: float = 1.0, opacity: Optional[float] = 0.5) -> None:
        self.visualize_batch(torch.ones_like(occupancy).unsqueeze(0).repeat(1, 3, 1, 1, 1), mask=occupancy.unsqueeze(0), T_world_object=T_world_object.unsqueeze(0) if T_world_object is not None else T_world_object, origin=origin.unsqueeze(0), pitch=pitch, opacity=opacity)
        # self.visualize_batch(occupancy.unsqueeze(0), T_world_object=T_world_object.unsqueeze(0) if T_world_object is not None else T_world_object, origin=origin.unsqueeze(0), pitch=pitch, opacity=opacity)

