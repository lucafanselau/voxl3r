from dataclasses import dataclass, field
from functools import partial
import os
from pathlib import Path
from typing import Generator, Optional, Tuple, List

import PIL
from PIL.ImageOps import exif_transpose
from beartype import beartype
from einops import rearrange
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split, Dataset
from jaxtyping import Float, Int, Bool, jaxtyped
from torch import Tensor
from utils.basic import get_default_device
import numpy as np
from tqdm import tqdm


from dataset import (
    SceneDataset,
    SceneDatasetConfig,
)
from experiments.mast3r import load_model, predict
from experiments.mast3r_baseline.module import (
    Mast3rBaselineConfig,
    Mast3rBaselineLightningModule,
)
from experiments.occ_chunk_dataset import OccChunkDataset, OccChunkDatasetConfig
from experiments.surface_net_3d.projection import project_voxel_grid_to_images_seperate
from extern.mast3r.dust3r.dust3r.utils.image import _resize_pil_image, load_images
from utils.chunking import (
    create_chunk,
    mesh_2_local_voxels,
)
from utils.data_parsing import load_yaml_munch
from utils.transformations import invert_pose
from multiprocessing import Pool

config = load_yaml_munch("./utils/config.yaml")


def update_camera_intrinsics(K, new):
    """
    new: tuple of (W_new, H_new)
    """
    K_old = K
    H_new_cropped, W_new_cropped = new
    W_old, H_old = 2 * K_old[0, 2], 2 * K_old[1, 2]

    S = max(W_old, H_old)
    long_edge_size = W_new_cropped.item()

    W_new, H_new = tuple(int(round(x * long_edge_size / S)) for x in (W_old, H_old))

    s_x = W_new / W_old
    s_y = H_new / H_old
    cx, cy = W_new // 2, H_new // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    delta_x_left, delta_y_top, _, _ = (cx - halfw, cy - halfh, cx + halfw, cy + halfh)

    return np.array(
        [
            [K_old[0][0] * s_x, 0.0, K_old[0][2] * s_x - delta_x_left],
            [0.0, K_old[1][1] * s_y, K_old[1][2] * s_y - delta_y_top],
            [0.0, 0.0, 1.0],
        ]
    )


@dataclass
class Mast3rChunkDatasetConfig(OccChunkDatasetConfig):
    mast3r_results_dir: Optional[str] = "mast3r_preprocessed"
    force_prepare_mast3r: bool = False


class Mast3rChunkDataset:
    def __init__(
        self,
        data_config: Mast3rChunkDatasetConfig,
        transform: Optional[callable] = None,
    ):
        self.data_config = data_config
        self.transform = transform
        self.base_dataset = OccChunkDataset(data_config, transform=None)

    # THIS NOW EXPECTS A BATCH
    def load_prepare(self, item):
        occ, data_dict = item
        image_names, transformations = data_dict["images"]
        image_paths = [
            str(Path("/", *Path(img).parts[Path(img).parts.index("mnt") :]))
            for names in image_names
            for img in names
        ]

        return image_paths, transformations

    def load_images(self, image_paths, size=512):
        mast3r_dicts = load_images(image_paths, size, verbose=False)
        images = torch.stack([d["img"] for d in mast3r_dicts]).squeeze(1)
        true_shapes = torch.stack(
            [torch.from_numpy(d["true_shape"]) for d in mast3r_dicts]
        ).squeeze(1)

        return images, true_shapes, image_paths

    @torch.no_grad()
    def process_chunk(self, batch, batch_idx, model):
        occ, data_dict = batch

        image_paths, transformations = self.load_prepare(batch)

        seq_len = len(data_dict["images"][0])
        B = occ.shape[0]

        def file_exists(idx):
            scene_name = data_dict["name"][idx]
            saving_path = (
                self.base_dataset.get_grid_path(scene_name)
                / self.data_config.mast3r_results_dir
            )
            if not saving_path.exists():
                saving_path.mkdir(parents=True)
            file_name = saving_path / (
                f"{batch_idx * B + idx}_"
                + str(data_dict["image_name_chunk"][idx])
                + ".pt"
            )

            return file_name.exists()

        if not self.data_config.force_prepare_mast3r and all(
            file_exists(idx) for idx in range(B)
        ):
            return

        images, true_shapes, _ = self.load_images(image_paths)

        for t in transformations:
            K = t["K"]
            for i in range(B):
                k = K[i]
                K_new = update_camera_intrinsics(k.numpy(), true_shapes[i].numpy())
                K[i] = torch.from_numpy(K_new)

        # images is B x 4, 3, ...

        img = rearrange(
            images, "(SEQ_LEN B) C H W -> SEQ_LEN B C H W", SEQ_LEN=seq_len
        ).to(get_default_device())
        shapes = rearrange(
            true_shapes, "(SEQ_LEN B) C -> SEQ_LEN B C", SEQ_LEN=seq_len
        ).to(get_default_device())

        res1, res2, dict1, dict2 = model.forward(
            rearrange(img[::2], "SEQ_LEN B C H W -> (SEQ_LEN B) C H W", B=B),
            rearrange(img[1::2], "SEQ_LEN B C H W -> (SEQ_LEN B) C H W", B=B),
            rearrange(shapes[::2], "SEQ_LEN B C -> (SEQ_LEN B) C", B=B),
            rearrange(shapes[1::2], "SEQ_LEN B C -> (SEQ_LEN B) C", B=B),
        )

        for idx in range(B):

            image_names = [str(Path(name).name) for name in image_paths[idx::B]]
            idx_res1 = {
                k + "_" + image_names[s * 2]: v[idx + (B * s)].detach().cpu()
                for k, v in res1.items()
                for s in range(seq_len // 2)
            }
            idx_res2 = {
                k + "_" + image_names[(s * 2 + 1)]: v[idx + (B * s)].detach().cpu()
                for k, v in res2.items()
                for s in range(seq_len // 2)
            }

            sample_data_dict = {
                k: v[idx] for k, v in data_dict.items() if k != "images"
            }

            master_chunk_dict = {
                "name": sample_data_dict["name"],
                "resolution": sample_data_dict["resolution"],
                "grid_size": sample_data_dict["grid_size"],
                "chunk_size": sample_data_dict["chunk_size"],
                "center": sample_data_dict["center"],
                "training_data": occ[idx],
                "image_name_chunk": sample_data_dict["image_name_chunk"],
                "images": (
                    [name for name in image_paths[idx::B]],
                    [
                        {k: v[idx] for k, v in transformations[i].items()}
                        for i in range(seq_len)
                    ],
                ),
                "pairwise_predictions": (idx_res1, idx_res2),
            }

            scene_name = data_dict["name"][idx]
            saving_path = (
                self.base_dataset.get_grid_path(scene_name)
                / self.data_config.mast3r_results_dir
            )

            file_name = saving_path / (
                f"{batch_idx * B + idx}_"
                + str(data_dict["image_name_chunk"][idx])
                + ".pt"
            )

            if not saving_path.exists():
                saving_path.mkdir(parents=True)

            torch.save(master_chunk_dict, file_name)

    @torch.no_grad()
    def prepare_data(self):
        self.base_dataset.prepare_data()

        model = Mast3rBaselineLightningModule(Mast3rBaselineConfig())
        model.eval()
        model.model.eval()
        model.model.to(get_default_device())

        batch_size = 8

        dataloader = DataLoader(
            self.base_dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
        )

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            total=len(self.base_dataset) // batch_size,
        ):
            self.process_chunk(batch, batch_idx, model)

        del dataloader
        del batch, batch_idx

        self.load_paths()

        # del model

        # if self.data_config.num_workers == 1:
        #     for idx in tqdm(range(len(self))):
        #         self.process_chunk(idx, model)
        # else:
        #     with Pool(self.data_config.num_workers) as p:
        #         args = [dict(idx=idx, model=model) for idx in range(len(self))]
        #         _ = list(tqdm(p.imap(self.process_chunk, args)), total=len(self))

    def load_paths(self):
        self.file_names = {}

        for scene_name in self.data_config.scenes:
            data_dir = (
                self.base_dataset.get_grid_path(scene_name)
                / self.data_config.mast3r_results_dir
            )
            if data_dir.exists():
                self.file_names[scene_name] = list(
                    [s for s in data_dir.iterdir() if s.is_file()]
                )

    def get_at_idx(self, idx: int):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")
        if os.path.getsize(file) < 0:  # 42219083:
            print(f"File {file} is empty. Skipping.")

        try:
            data = torch.load(file)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
        occupancy_grid = data["training_data"]
        return occupancy_grid, data

    def __getitem__(self, idx):

        result = self.get_at_idx(idx)

        if self.transform is not None:
            result = self.transform(result, idx)

        return result

    def __len__(self):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )
        return sum([len(self.file_names[scene_name]) for scene_name in self.file_names])
