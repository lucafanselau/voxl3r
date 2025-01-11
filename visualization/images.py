from typing import Optional
from jaxtyping import jaxtyped, Bool, Float
import numpy as np
from torch import Tensor

from utils.transformations import extract_rot_trans, invert_pose
from . import base

import pyvista as pv

class Config(base.Config):
    pass

# write 16 different colors
colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "pink", "brown", "grey", "black", "white", "gold", "silver", "maroon"]


class Visualizer(base.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)
        
    def add_from_image_dict(self, image_dict: dict, base_coordinate_frame: Optional[Float[Tensor, "4 4"]] = None) -> None:
        """

        Args:
            image_dict (dict): image dict of image that is to be visualized
            base_coordinate_frame (Optional[Tensor[float]], optional): additional transformation to apply to the image. Defaults to None.
        """
        camera_params_list = image_dict["cameras"]
        image_paths = image_dict["images"]
        
        for i, (image_path, camera_params) in enumerate(zip(image_paths, camera_params_list)):
            texture = pv.read_texture(image_path)
            _, _, T_wc = invert_pose(*extract_rot_trans(camera_params["T_cw"]))
            transform = base_coordinate_frame@T_wc if base_coordinate_frame is not None else T_wc
            self.add_image(texture, transform, camera_params["K"], camera_params["height"], camera_params["width"], highlight=i)
        
    def add_image(self, texture: pv.Texture, transform: base.Transformation, intrinsics: Float[Tensor, "3 3"], height: int, width: int, highlight: int = 0) -> None:
        """
        Add an image to the visualizer.

        Args:
            image (pv.Texture): The image to add.
            T_wc (np.ndarray): The transformation matrix from camera to world.
            intrinsics (np.ndarray): The intrinsics matrix.
        """
        c_point = pv.PolyData(transform[:3, 3].reshape(1, 3))
        self.plotter.add_mesh(
            c_point, point_size=10, render_points_as_spheres=True,
            # red if highlight else grey
            color=colors[highlight if highlight < len(colors) else 0]
        )
        
        T_cw = np.linalg.inv(transform)
        T_wc = transform

        """
        # Draw image plane
        corners_cam = self.get_camera_corners(
            T_cw, height, width,  plane_distance + (offsets[-1] if len(offsets) > 0 else 0)
        )
        
        for corner in corners_cam:
            corner = T_wc[:3,:3] @ corner + T_wc[:3, 3].flatten()
            line_points = np.array([T_wc[:3, 3].flatten(), corner])
            line = pv.lines_from_points(line_points)
            self.plotter.add_mesh(line, color="red" if i == 0 else "black", line_width=4)
        """


        plane = self.create_image_plane(transform, intrinsics, height, width, plane_distance=0.1)
        self.plotter.add_mesh(plane, texture=texture)
        
    def create_image_plane(self, T_wc, intrinsics, height, width, plane_distance):
        """
        Create a PyVista plane representing the image in 3D space with correct orientation.

        Args:
            T_wc (dict): The transformation matrix from camera to world.
            plane_distance (float): Distance from the camera center to the plane.

        Returns:
            pv.PolyData: The plane positioned in 3D space.
        """
        corners_cam = self.get_camera_corners(intrinsics, height, width, plane_distance)

        # Compute the center of the plane in camera coordinates
        center_cam = corners_cam.mean(axis=0)  # Shape: [3,]

        # The plane's normal vector in camera coordinates
        direction_cam = np.array(
            [0, 0, 1]
        )  # Assuming the plane is facing along the positive Z-axis

        # The size of the plane along i and j axes (in camera coordinates)
        i_size = np.linalg.norm(corners_cam[1] - corners_cam[0])  # Width of the plane
        j_size = np.linalg.norm(corners_cam[3] - corners_cam[0])  # Height of the plane

        # Create the plane in camera coordinates
        plane = pv.Plane(
            center=center_cam,
            direction=direction_cam,
            i_size=i_size,
            j_size=j_size,
            i_resolution=1,
            j_resolution=1,
        )

        # Compute the rotation matrix to align the plane's local axes with the camera's axes
        # Rotate the plane around the X-axis by 180 degrees to flip the Y-axis
        plane.rotate_x(180, point=center_cam, inplace=True)

        # Apply the transformation
        plane.transform(T_wc)

        # Assign texture coordinates (UV mapping)
        # The plane's texture coordinates need to be adjusted because of the flip
        plane.texture_map_to_plane(inplace=False)

        return plane
    
    def get_camera_corners(self, intrinsics, heigth, width, plane_distance=1.0):
        """
        Get the 3D coordinates of the four image corners in camera coordinates at a given plane distance.

        Args:
            cam_params (dict): Camera parameters containing 'K' (3x3 intrinsic matrix).
            plane_distance (float): Distance of the image plane from the camera center along the Z-axis.

        Returns:
            np.ndarray: Array of shape (4, 3) with the 3D coordinates of the four corners
                        in camera coordinates at the specified plane distance.
        """
        
        # Define the four corners of the image in pixel coordinates
        pixel_corners = np.array(
            [
                [0, 0],  # Top-left corner
                [width, 0],  # Top-right corner
                [width, heigth],  # Bottom-right corner
                [0, heigth],  # Bottom-left corner
            ]
        )  # Shape: [4, 2]

        # Convert pixel coordinates to homogeneous coordinates
        homogeneous_pixel_corners = np.hstack(
            [pixel_corners, np.ones((4, 1))]
        )  # Shape: [4, 3]

        # Compute the inverse of the intrinsic matrix K
        K_inv = np.linalg.inv(intrinsics)

        # Convert to normalized camera coordinates
        corners_cam = (K_inv @ homogeneous_pixel_corners.T).T  # Shape: [4, 3]

        # Scale the normalized coordinates to have Z = plane_distance
        corners_cam *= plane_distance / corners_cam[:, 2:3]

        return corners_cam

