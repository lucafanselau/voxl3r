import os
import csv

import numpy as np

# Define constants
DATA_PREPROCESSING_DIR = "data_preprocessing"
CAMERA_EXTRINSICS_FILENAME = "camera_extrinsics.csv"
IMAGES_PATHNAME = "colmap/images.txt"
HEADER_LINES_TO_SKIP = 4
CSV_HEADER = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']

def create_camera_extrinsics_csv(scene, camera="dslr", data_dir="data", verbose=True):
    # Define input and output file paths
    input_file = os.path.join(data_dir, scene, camera, IMAGES_PATHNAME)
    output_dir = os.path.join(data_dir, scene, camera, DATA_PREPROCESSING_DIR)
    output_file = os.path.join(output_dir, CAMERA_EXTRINSICS_FILENAME)
    
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        csv_writer = csv.writer(f_out)
        
        # Write header to the CSV file
        header = CSV_HEADER
        csv_writer.writerow(header)
        
        # Skip the first 4 header lines in the input file
        for _ in range(HEADER_LINES_TO_SKIP):
            next(f_in)
        
        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty or comment lines
            
            if line_number % 2 == 1:
                # Process odd-numbered lines (camera extrinsic data)
                parts = line.strip().split()
                if len(parts) >= 10:
                    csv_writer.writerow(parts[:10])
                else:
                    if verbose:
                        print(f"Warning: Line {line_number + 4} does not have enough data fields.")
            else:
                # Skip even-numbered lines
                continue
    if verbose:
        print(f"Camera extrinsics saved to {output_file}")

def get_camera_params(camera_path):
        with open(camera_path, 'r') as f:
            camera_params = f.readlines()[3].split()
        MODEL, WIDTH, HEIGHT = camera_params[1], camera_params[2], camera_params[3]
        
        if MODEL == 'PINHOLE':
            fx, fy, cx, cy = [float(p) for p in camera_params[4:]]
            K = np.array([[fx, 0,  cx], [0,  fy, cy], [0,   0,  1]], dtype=np.float64)
            dist_coeffs = np.zeros((4,), dtype=np.float64)
        elif MODEL == 'OPENCV':
            fx, fy, cx, cy, k1, k2, p1, p2 = [float(p) for p in camera_params[4:]]
            K = np.array([[fx, 0,  cx], [0,  fy, cy], [0,   0,  1]])
            dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64).T
        else:
            raise NotImplementedError(f"Camera model {MODEL} is not implemented.")
        
        return {
        'K': K,
        'dist_coeffs': dist_coeffs,
        'width': float(WIDTH),
        'height': float(HEIGHT)
        }
        
def read_depth_bin(file_path, image_height, image_width):
    """
    Reads a binary depth file containing 16-bit depth images and returns a list of depth images.

    Args:
        file_path (str): Path to the `.bin` file.
        image_height (int): Height of each depth image.
        image_width (int): Width of each depth image.

    Returns:
        List[np.ndarray]: List of depth images as numpy arrays.
    """
    # Calculate the number of pixels in each image
    num_pixels_per_image = image_height * image_width

    with open(file_path, 'rb') as f:
        data = f.read()

    # Ensure the data can be reshaped into a sequence of images
    num_images = data.size // num_pixels_per_image
    if data.size % num_pixels_per_image != 0:
        raise ValueError("The data size is not a multiple of the image dimensions; check the file format.")

    # Reshape the data into (num_images, image_height, image_width)
    depth_images = data.reshape((num_images, image_height, image_width))

    return depth_images
    
