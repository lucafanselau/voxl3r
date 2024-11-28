from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from einops import rearrange
import numpy as np
import pyvista as pv
import torch
from jaxtyping import Float
from torch import Tensor

from dataset import SceneDataset


def calculate_average_color(
    voxels: Float[Tensor, "batch seq_len 3 depth height width"], fill_value=-1
) -> Float[Tensor, "batch 3 depth height width"]:
    # Create mask for valid values (not -1)
    mask = voxels != fill_value

    # Count number of valid values per voxel location
    # Sum across images and channels, then divide by 3 to get count per voxel
    valid_counts = torch.sum(torch.sum(mask, dim=2), dim=1) / 3

    # Set invalid values to 0 for averaging
    voxels = torch.where(mask, voxels, torch.tensor(0.0, device=voxels.device))

    # Sum across images dimension
    color_sums = torch.sum(voxels, dim=1)

    # Avoid division by zero by masking where count is 0
    valid_mask = valid_counts > 0

    # Initialize output tensor with zeros
    output = torch.zeros_like(color_sums)

    # Calculate average only for valid voxels
    output = torch.where(
        valid_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1),
        color_sums / valid_counts.unsqueeze(1).repeat(1, 3, 1, 1, 1),
        output,
    )

    return output


@dataclass
class VoxelVisualizerConfig:
    """Configuration for voxel grid visualization."""

    # Visualization settings
    opacity: float = 0.5
    show_edges: bool = True
    edge_color: str = "black"
    cmap: str = "viridis"
    background_color: str = "white"
    window_size: Tuple[int, int] = (1024, 768)

    # Grid settings
    grid_spacing: float = 1.0
    grid_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Camera settings
    camera_position: Optional[Tuple[float, float, float]] = None
    view_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Multi-grid layout
    grid_arrangement: Optional[Tuple[int, int]] = None  # Rows, Cols for multiple grids
    spacing_factor: float = 1.2  # Space between grids

    # Add new visualization options
    clip_plane_normal: Optional[Tuple[float, float, float]] = None
    clip_plane_origin: Optional[Tuple[float, float, float]] = None
    use_opacity_array: bool = False
    opacity_threshold: float = 0.5  # For thresholding when using opacity arrays

    # Add new parameters for labels
    show_labels: bool = False
    label_size: int = 12
    label_color: str = "black"
    label_offset: Tuple[float, float, float] = (0.0, 0.0, -1.0)


class VoxelGridVisualizer:
    def __init__(self, config: VoxelVisualizerConfig = VoxelVisualizerConfig()):
        """Initialize the visualizer with given configuration."""
        self.config = config
        self.plotter = pv.Plotter(
            # window_size=self.config.window_size,
        )
        self.plotter.background_color = self.config.background_color

    def _tensor_to_numpy(
        self, tensor: Float[Tensor, "batch channels depth height width"]
    ) -> np.ndarray:
        """Convert PyTorch tensor to numpy array and ensure correct shape."""
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        return tensor

    def _create_voxel_grid(
        self, values: np.ndarray, origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> pv.StructuredGrid:
        """Create a PyVista structured grid from voxel values.

        The grid is created with one more point than cells in each dimension,
        since points define the corners of cells.
        """
        X_shape, Y_shape, Z_shape = values.shape[-3:]

        # Create coordinate arrays with one more point than cells in each dimension
        x = np.arange(X_shape + 1) * self.config.grid_spacing + origin[0]
        y = np.arange(Y_shape + 1) * self.config.grid_spacing + origin[1]
        z = np.arange(Z_shape + 1) * self.config.grid_spacing + origin[2]

        # Create mesh grid of points
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)

        return grid

    def visualize_single_grid(
        self,
        voxels: Float[Tensor, "channels depth height width"],
        mask: Optional[Float[Tensor, "depth height width"]] = None,
        show: bool = True,
    ) -> pv.StructuredGrid:
        """Visualize a single voxel grid."""
        voxels = self._tensor_to_numpy(voxels)

        # Create grid
        grid = self._create_voxel_grid(voxels)

        if voxels.shape[0] == 3:  # RGB data
            rgb = np.moveaxis(voxels, 0, -1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            # Use cell_data instead of point_data since we want to color the cells
            grid.cell_data["RGB"] = rgb.reshape(-1, 3)

            # Apply masking if provided
            if mask is not None:
                mask = self._tensor_to_numpy(mask)
                grid.cell_data["mask"] = mask.flatten()
                grid = grid.threshold(0.5, scalars="mask")

            self.plotter.add_mesh(
                grid,
                rgb=True,
                show_edges=self.config.show_edges,
                edge_color=self.config.edge_color,
            )
        else:
            # Use cell_data for scalar values as well
            grid.cell_data["values"] = voxels[0].flatten()

            # Apply masking if provided
            if mask is not None:
                mask = self._tensor_to_numpy(mask)
                grid.cell_data["mask"] = mask.flatten()
                grid = grid.threshold(0.5, scalars="mask")

            self.plotter.add_mesh(
                grid,
                scalars="values",
                cmap=self.config.cmap,
                show_edges=self.config.show_edges,
                edge_color=self.config.edge_color,
            )

        if show:
            self.show()

        return grid

    def visualize_batch(
        self,
        voxels: Float[Tensor, "batch channels depth height width"],
        mask: Optional[Float[Tensor, "batch depth height width"]] = None,
        labels: Optional[List[str]] = None,
        show: bool = True,
    ) -> List[pv.StructuredGrid]:
        """Visualize a batch of voxel grids with optional labels."""
        voxels = self._tensor_to_numpy(voxels)
        batch_size = voxels.shape[0]

        # Determine grid arrangement
        if self.config.grid_arrangement is None:
            rows = int(np.ceil(np.sqrt(batch_size)))
            cols = int(np.ceil(batch_size / rows))
        else:
            rows, cols = self.config.grid_arrangement

        grids = []
        for i in range(batch_size):
            # Calculate grid position
            row = i // cols
            col = i % cols

            # Calculate offset for this grid
            offset = (
                col
                * self.config.grid_spacing
                * voxels.shape[-1]
                * self.config.spacing_factor,
                row
                * self.config.grid_spacing
                * voxels.shape[-2]
                * self.config.spacing_factor,
                0,
            )

            # Create and add grid
            current_mask = mask[i] if mask is not None else None
            grid = self._create_voxel_grid(voxels[i], origin=offset)

            if voxels.shape[1] == 3:  # RGB
                rgb = np.moveaxis(voxels[i], 0, -1)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                grid.cell_data["RGB"] = rgb.reshape(-1, 3)

                if current_mask is not None:
                    current_mask = self._tensor_to_numpy(current_mask)
                    grid.cell_data["mask"] = current_mask.flatten()
                    grid = grid.threshold(0.5, scalars="mask")

                self.plotter.add_mesh(
                    grid,
                    rgb=True,
                    show_edges=self.config.show_edges,
                    edge_color=self.config.edge_color,
                )
            else:
                grid.cell_data["values"] = voxels[i, 0].flatten()

                if current_mask is not None:
                    current_mask = self._tensor_to_numpy(current_mask)
                    grid.cell_data["mask"] = current_mask.flatten()
                    grid = grid.threshold(0.5, scalars="mask")

                self.plotter.add_mesh(
                    grid,
                    scalars="values",
                    cmap=self.config.cmap,
                    show_edges=self.config.show_edges,
                    edge_color=self.config.edge_color,
                )

            # Add label if provided using Text3D
            if self.config.show_labels and labels is not None and i < len(labels):
                # Calculate label position relative to the grid
                grid_width = voxels.shape[-1] * self.config.grid_spacing
                grid_height = voxels.shape[-2] * self.config.grid_spacing

                # Create position for the 3D text
                text_pos = (
                    offset[0] + grid_width / 2,  # Center of grid in x
                    offset[1] - (grid_height * 0.1),  # Slightly above grid
                    offset[2],  # Same z level
                )

                # Create 3D text
                text_3d = pv.Text3D(
                    labels[i],
                    depth=0.1,  # Small depth for subtle 3D effect
                    height=grid_height * 0.15,  # Scale height relative to grid
                    center=text_pos,  # Position the text
                )

                # Add the 3D text to the scene
                self.plotter.add_mesh(
                    text_3d,
                    color=self.config.label_color,
                )

            grids.append(grid)

        if show:
            self.show()

        return grids

    def show(self):
        """Display the visualization."""
        if self.config.camera_position is not None:
            self.plotter.camera_position = self.config.camera_position
            self.plotter.camera.up = self.config.view_up

        self.plotter.show()

    def save_screenshot(self, filename: str):
        """Save the current view to an image file."""
        self.plotter.screenshot(filename)

    def close(self):
        """Close the plotter."""
        self.plotter.close()


def visualize_voxel_grids(
    voxels: Float[Tensor, "batch channels depth height width"],
    config: Optional[VoxelVisualizerConfig] = None,
    save_path: Optional[str] = None,
    mask: Optional[Float[Tensor, "batch depth height width"]] = None,
    labels: Optional[List[str]] = None,
) -> None:
    """Convenience function to quickly visualize voxel grids."""
    if config is None:
        config = VoxelVisualizerConfig()

    visualizer = VoxelGridVisualizer(config)

    if len(voxels.shape) == 4:  # Single grid
        visualizer.visualize_single_grid(voxels, mask=mask)
    else:  # Batch of grids
        visualizer.visualize_batch(voxels, mask=mask, labels=labels)

    if save_path is not None:
        visualizer.save_screenshot(save_path)

    visualizer.close()


# main function
if __name__ == "__main__":
    from experiments.surface_net_3d.data import (
        SurfaceNet3DDataConfig,
        VoxelGridDataset,
        UnwrapVoxelGridTransform,
    )

    data_dir = "/home/luca/mnt/data/scannetpp/data"
    base_dataset = SceneDataset(data_dir)

    # create only a voxel grid dataset
    data_config = SurfaceNet3DDataConfig(data_dir=data_dir)
    grid_dataset = VoxelGridDataset(
        base_dataset, data_config, transform=UnwrapVoxelGridTransform()
    )
    grid_dataset.prepare_data()

    features, occupancy = grid_dataset[0]
    # add fake batch dimension
    features = features.unsqueeze(0)
    occupancy = occupancy.unsqueeze(0)

    # Visualize features (first sample from batch)
    vis_config = VoxelVisualizerConfig(
        opacity=1,
        show_edges=True,
        cmap="viridis",
        window_size=(1200, 800),
        camera_position=(100, 100, 100),
    )

    # Visualize feature channels
    # channels is 48: 16 images (seq_len) * 3 (rgb)
    # we want to visualize the average color for each voxel
    # first resize features to indicate the images (batch images 3 depth height width)
    features_color = calculate_average_color(features)

    # Create mask from occupancy (threshold at 0.5)
    occupancy_mask = (occupancy != 0).squeeze(1).cpu()  # Remove channel dimension

    # Create visualization tensor with masked colors and occupancy
    visualization = torch.cat(
        [features_color, occupancy.detach().cpu().repeat(1, 3, 1, 1, 1)], dim=0
    )

    # Create combined mask tensor (True for features where occupancy > 0.5, True for all occupancy visualization)
    mask = torch.cat(
        [
            occupancy_mask,  # Mask for feature visualization
            occupancy_mask,  # Mask for occupancy visualization
        ],
        dim=0,
    )

    print("Visualizing feature channels...")
    visualize_voxel_grids(
        visualization,
        config=vis_config,
        mask=mask,
        save_path="feature_channels.png",
    )
