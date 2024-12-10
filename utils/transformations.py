import cv2
import numpy as np
import torch
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R


def invert_pose_batched(
    R_cw: Float[torch.Tensor, "B 3 3"], t_cw: Float[torch.Tensor, "B 3 1"]
) -> Float[torch.Tensor, "B 4 4"]:

    B, _, _ = R_cw.shape
    t_cw = t_cw.reshape(-1, 3, 1)
    R_wc = torch.transpose(R_cw, 1, 2)
    t_wc = -1.0 * torch.matmul(R_wc, t_cw)

    T_wc = torch.eye(4, 4).unsqueeze(0).repeat(B, 1, 1)
    T_wc[:, :3, :3] = R_wc
    T_wc[:, :3, 3:] = t_wc

    return T_wc

def extract_rot_trans(T_cw):
    return T_cw[:3, :3], T_cw[:3, 3]

def from_rot_trans(R_cw, t_cw):
    T_cw = torch.eye(4, 4)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3:] = t_cw
    return T_cw

def from_rot_trans_batched(R_cw, t_cw):
    B, _, _ = R_cw.shape
    T_cw = torch.eye(4, 4).unsqueeze(0).repeat(B, 1, 1)
    T_cw[:, :3, :3] = R_cw
    T_cw[:, :3, 3:] = t_cw
    return T_cw

def invert_pose(R_cw, t_cw):
    """
    Inverts the camera pose from camera-to-world to world-to-camera coordinates.

    Args:
        R_cw (ndarray): Rotation matrix from camera to world coordinates (shape: [3, 3]).
        t_cw (ndarray): Translation vector from camera to world coordinates (shape: [3, 1]).

    Returns:
        R_wc (ndarray): Rotation matrix from world to camera coordinates (shape: [3, 3]).
        t_wc (ndarray): Translation vector from world to camera coordinates (shape: [3, 1]).
    """
    t_cw = t_cw.reshape(3, 1)
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    T_wc = np.vstack((np.hstack((R_wc, t_wc)), np.array([0, 0, 0, 1])))
    return R_wc, t_wc, T_wc


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion into a rotation matrix.

    Args:
        qw, qx, qy, qz (float): Quaternion components.

    Returns:
        R_matrix (ndarray): Rotation matrix (shape: [3, 3]).
    """
    r = R.from_quat([qx, qy, qz, qw])
    R_matrix = r.as_matrix()
    return R_matrix


def project_image_plane_single_camera(c_params, points):
    """
    Projects multiple 3D world points into the image plane.

    Args:
        points (ndarray): 3D points in world coordinates (shape: [3, N]).
        K (ndarray): Intrinsic camera matrix (shape: [3, 3]).
        R_cw (ndarray): Rotation matrix from world to camera coordinates (shape: [3, 3]).
        t_cw (ndarray): Translation vector from camera center to world origin (shape: [3, 1]).

    Returns:
        uvs (ndarray): Projected pixel coordinates (shape: [2, M]).
        valid_indices (ndarray): Indices of points that are in front of the camera.
    """
    points = points.reshape(3, -1)
    p_camera = c_params["R_cw"] @ points + c_params["t_cw"]  # Shape: [3, N]

    in_front = p_camera[2, :] > 0  # Boolean array of shape [N]
    p_camera = p_camera[:, in_front]  # Shape: [3, M], where M <= N

    if p_camera.shape[1] == 0:
        return None, None  # No points in front of the camera

    homogeneous_coord = p_camera / p_camera[2, :]
    pixel_coords = c_params["K"] @ homogeneous_coord  # Shape: [3, M]

    u = pixel_coords[0, :]
    v = pixel_coords[1, :]
    uvs = np.vstack((u, v))  # Shape: [2, M]

    valid_indices = np.where(in_front)[0]

    return uvs, valid_indices


def project_image_plane(c_params, points):
    """
    Projects multiple 3D world points into the image plane.

    Args:
        c_params_list (list): List of camera parameters dictionaries.
        points (ndarray): 3D points in world coordinates (shape: [3, N]).
    Returns:
        uvs (list[ndarray]): List of projected pixel coordinates (shape: [2, M]).
        valid_indices (list[ndarray]): List for indices of points that are in front of the camera.
    """
    uvs = {}
    valid_indices = {}

    for key in c_params.keys():
        uv, valid_idx = project_image_plane_single_camera(c_params[key], points)
        uvs[key] = uv
        valid_indices[key] = valid_idx
    return uvs, valid_indices


def undistort_image(image, K, dist_coeffs, alpha=0):
    """
    Undistort a single image using OpenCV's undistort function with optimized camera matrix.

    Args:
        image (numpy.ndarray): Image array in HxWxC format (uint8 or uint16).
        K (numpy.ndarray): Intrinsic camera matrix (3x3).
        dist_coeffs (numpy.ndarray): Distortion coefficients (e.g., [k1, k2, p1, p2, k3]).
        alpha (float): Free scaling parameter between 0 and 1.

    Returns:
        numpy.ndarray: Undistorted image, optionally cropped to remove black borders.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        is_color = False
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Color image
        is_color = True
    else:
        raise ValueError(
            "Unsupported image format. Image must be either grayscale or RGB."
        )

    # Get image dimensions
    h, w = image.shape[:2]

    # Compute the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        K, dist_coeffs, (w, h), alpha, (w, h)
    )

    # Undistort the image
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_camera_matrix)

    # Optionally crop the image to the valid ROI to remove black borders
    x, y, w_roi, h_roi = roi
    undistorted_image = undistorted_image[y : y + h_roi, x : x + w_roi]

    return undistorted_image
