import collections
import json
import os
import csv

from munch import Munch
import numpy as np
import pandas as pd
import yaml

from utils.transformations import invert_pose, quaternion_to_rotation_matrix

# Define constants
DATA_PREPROCESSING_DIR = "data_preprocessing"
CAMERA_EXTRINSICS_FILENAME = "camera_extrinsics.csv"
IMAGES_PATHNAME = "colmap/images.txt"
HEADER_LINES_TO_SKIP = 4
CSV_HEADER = ["IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME"]


def create_camera_extrinsics_csv(camera_path, verbose=True):
    # Define input and output file paths
    input_file = os.path.join(camera_path, IMAGES_PATHNAME)
    output_dir = os.path.join(camera_path, DATA_PREPROCESSING_DIR)
    output_file = os.path.join(output_dir, CAMERA_EXTRINSICS_FILENAME)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as f_in, open(output_file, "w", newline="") as f_out:
        csv_writer = csv.writer(f_out)

        # Write header to the CSV file
        header = CSV_HEADER
        csv_writer.writerow(header)

        # Skip the first 4 header lines in the input file
        for _ in range(HEADER_LINES_TO_SKIP):
            next(f_in)

        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty or comment lines

            if line_number % 2 == 1:
                # Process odd-numbered lines (camera extrinsic data)
                parts = line.strip().split()
                if len(parts) >= 10:
                    csv_writer.writerow(parts[:10])
                else:
                    if verbose:
                        print(
                            f"Warning: Line {line_number + 4} does not have enough data fields."
                        )
            else:
                # Skip even-numbered lines
                continue
    if verbose:
        print(f"Camera extrinsics saved to {output_file}")


def get_camera_intrisics(camera_path, camera):
    if camera == "dslr":
        config_dslr = load_yaml_munch(camera_path / "nerfstudio" / "transforms_undistorted.json")
        
        MODEL, WIDTH, HEIGHT = config_dslr["camera_model"], config_dslr["w"], config_dslr["h"]
        fx, fy, cx, cy = config_dslr["fl_x"], config_dslr["fl_y"], config_dslr["cx"], config_dslr["cy"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([config_dslr["k1"], config_dslr["k2"], config_dslr["k3"], config_dslr["k4"]], dtype=np.float64)
    else:
        with open(camera_path / "colmap" / "cameras.txt", "r") as f:
            camera_params = f.readlines()[3].split()
        MODEL, WIDTH, HEIGHT = camera_params[1], camera_params[2], camera_params[3]

        if MODEL == "PINHOLE":
            fx, fy, cx, cy = [float(p) for p in camera_params[4:]]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros((4,), dtype=np.float64)
        elif MODEL == "OPENCV":
            fx, fy, cx, cy, k1, k2, p1, p2 = [float(p) for p in camera_params[4:]]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64).T
        else:
            raise NotImplementedError(f"Camera model {MODEL} is not implemented.")

    return {
        "K": K,
        "dist_coeffs": dist_coeffs,
        "width": float(WIDTH),
        "height": float(HEIGHT),
    }


def get_image_names_with_extrinsics(camera_path):
    camera_extrinsics_file = (
        camera_path / DATA_PREPROCESSING_DIR / CAMERA_EXTRINSICS_FILENAME
    )
    if not camera_extrinsics_file.exists():
        create_camera_extrinsics_csv(camera_path)
    df = pd.read_csv(camera_extrinsics_file)
    return df["NAME"].tolist()


def get_camera_params(scene_path, camera, image_name, seq_len):
    camera_path = scene_path / camera
    camera_extrinsics_file = (
        camera_path / DATA_PREPROCESSING_DIR / CAMERA_EXTRINSICS_FILENAME
    )

    camera_intrinsics = get_camera_intrisics(camera_path, camera)

    if not camera_extrinsics_file.exists():
        create_camera_extrinsics_csv(camera_path)

    df = pd.read_csv(camera_extrinsics_file)

    if isinstance(image_name, str):
        image_idx = df["NAME"].tolist().index(image_name)
        image_names = df["NAME"].tolist()[image_idx : image_idx + seq_len]
    elif isinstance(image_name, list):
        image_names = image_name
    elif image_name is None:
        image_names = df["NAME"].tolist()
    else:
        raise TypeError("image_name must be a string, list of strings, or None.")

    df_filtered = df[df["NAME"].isin(image_names)]

    missing_images = set(image_names) - set(df_filtered["NAME"])
    if missing_images:
        raise ValueError(
            f"For scene {scene_path.name} - no camera extrinsics found for image(s): {', '.join(missing_images)}"
        )

    params_dict = {}
    for _, row in df_filtered.iterrows():
        name = row["NAME"]
        qw, qx, qy, qz = row[["QW", "QX", "QY", "QZ"]].values
        tx, ty, tz = row[["TX", "TY", "TZ"]].values
        t_cw = np.array([[tx], [ty], [tz]])
        R_cw = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        T_cw = np.vstack((np.hstack((R_cw, t_cw)), np.array([0, 0, 0, 1])))
        R_wc, t_wc, T_wc = invert_pose(R_cw, t_cw)
        params_dict[name] = {
            "R_cw": R_cw,
            "t_cw": t_cw,
            "T_cw": T_cw,
            "T_wc": T_wc,
            "K": camera_intrinsics["K"],
            "dist_coeffs": camera_intrinsics["dist_coeffs"],
            "width": camera_intrinsics["width"],
            "height": camera_intrinsics["height"],
        }   
    
    if camera == "dslr":
        train_images = load_yaml_munch(camera_path / "train_test_lists.json")["train"]
        params_dict = {k: v for k, v in params_dict.items() if k in train_images}
        
    params_dict = collections.OrderedDict(sorted(params_dict.items()))
    return params_dict


def get_vertices_labels(scene_path):
    with open(scene_path / "scans" / "segments_anno.json", "r") as file:
        annotations = json.load(file)
    labels = {}
    for object in annotations["segGroups"]:
        if object["label"] not in labels.keys():
            labels[object["label"]] = object["segments"]
        else:
            labels[object["label"]].extend(object["segments"])
    return labels


def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)
