from typing import Optional, Tuple
from beartype import beartype
from jaxtyping import jaxtyped, Bool, Float, Int
import pyvista as pv
import numpy as np
from torch import Tensor
from . import base
from einops import rearrange
from datasets.chunk import occupancy
from utils.transformations import from_rot_trans, invert_pose, extract_rot_trans
import torch

class Config(base.Config):
    pass

class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)

    def add_from_occupancy_dict(self, occupancy_dict: dict) -> None:

        data = occupancy_dict

        occupancy = data["occupancy_grid"]
        center = data["center"]
        # T_center = from_rot_trans(torch.eye(3, 3), center)

        T_cw = data["images"][1][0]["T_cw"]
        _, _, T_wc = invert_pose(*extract_rot_trans(T_cw))

        # T_world_object = T_wc @ T_center 

        # occupancy = rearrange(occupancy, "1 X Y Z -> 1 Y X Z")

        self.add_occupancy(occupancy.int(), torch.Tensor(T_wc), pitch=data["resolution"], origin=torch.Tensor(center))

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
                grid.cell_data["RGB"] = rgb.reshape(-1, 3)

                if current_mask is not None:
                    current_mask = current_mask.cpu().numpy()
                    grid.cell_data["mask"] = current_mask.flatten()
                    grid = grid.threshold(0.5, scalars="mask")

                if T_world_object is not None:
                    grid.transform(T_world_object[i].cpu().numpy(), inplace=True)

                self.plotter.add_mesh(
                    grid,
                    rgb=True,
                )
            else:
                grid.cell_data["values"] = voxels[i, 0].flatten()

                if current_mask is not None:
                    current_mask = current_mask.cpu().numpy()
                    grid.cell_data["mask"] = current_mask.flatten()
                    grid = grid.threshold(0.5, scalars="mask")

                if T_world_object is not None:
                    grid.transform(T_world_object[i].cpu().numpy(), inplace=True)

                self.plotter.add_mesh(
                    grid,
                    scalars="values",
                )

            # Add outline box
            if T_world_object is not None:
                outline.transform(T_world_object[i].cpu().numpy(), inplace=True)
            self._add_outline(outline) 



    @jaxtyped(typechecker=beartype)
    def add_occupancy(self, occupancy: Int[Tensor, "1 X Y Z"], T_world_object: Optional[base.Transformation] = None, origin: Float[Tensor, "3"] = torch.zeros(3), pitch: float = 1.0) -> None:
        self.visualize_batch(torch.ones_like(occupancy).unsqueeze(0), mask=occupancy.unsqueeze(0), T_world_object=T_world_object.unsqueeze(0), origin=origin.unsqueeze(0), pitch=pitch)
