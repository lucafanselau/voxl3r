from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Tuple
from typing_extensions import TypedDict

from einops import rearrange
import torch
from torch.utils.data import DataLoader
from jaxtyping import Float
from datasets.chunk.base import ChunkBaseDataset, ChunkBaseDatasetConfig
from datasets.chunk import image
from utils.basic import get_default_device
import numpy as np
from tqdm import tqdm


from datasets import scene
# NOTE: we should probably not import from experiments here
from experiments.mast3r_baseline.module import (
    Mast3rBaselineConfig,
    Mast3rBaselineLightningModule,
)
from extern.mast3r.dust3r.dust3r.utils.image import load_images
from utils.data_parsing import load_yaml_munch

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


class Config(ChunkBaseDatasetConfig):
    folder_name_mast3r: Optional[str] = "mast3r_preprocessed"
    batch_size_mast3r: int = 8
    force_prepare_mast3r: bool = False

class Mast3rOutput(TypedDict):
    pts3d: Float[torch.Tensor, "B W H C"]
    conf: Float[torch.Tensor, "B W H"]
    desc: Float[torch.Tensor, "B W H D=24"]
    desc_conf: Float[torch.Tensor, "B W H"]

class Output(TypedDict):
    scene_name: str
    file_name: str
    image_name_chunk: str
    pairwise_predictions: Tuple[Mast3rOutput, Mast3rOutput]


class Dataset(ChunkBaseDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: scene.Dataset,
        image_dataset: image.Dataset,
    ):
        super(Dataset, self).__init__(data_config, base_dataset)
        self.image_dataset = image_dataset
        self.data_config = data_config

    # THIS NOW EXPECTS A BATCH
    def load_prepare(self, item):
        data_dict = item
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

    def get_chunk_dir(self, scene_name):
        image_data_config = self.image_dataset.data_config

        base_folder_name = f"seq_len_{image_data_config.seq_len}_furthest_{image_data_config.with_furthest_displacement}_center_{image_data_config.center_point}"

        path = (
            self.get_saving_path(scene_name)
            / self.data_config.folder_name_mast3r
            / base_folder_name
        )

        return path

    @torch.no_grad()
    def process_chunk(self, batch, batch_idx, model):
        data_dict = batch

        image_paths, transformations = self.load_prepare(batch)

        seq_len = len(data_dict["images"][0])
        # check batch size
        B = len(data_dict["images"][0][0])

        def file_exists(idx):
            scene_name = data_dict["scene_name"][idx]
            saving_dir = self.get_chunk_dir(scene_name)
            if not saving_dir.exists():
                saving_dir.mkdir(parents=True)
            file_name = saving_dir / (data_dict["file_name"][idx])

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

            master_chunk_dict = {
                "scene_name": data_dict["scene_name"][idx],
                "file_name": data_dict["file_name"][idx],
                "image_name_chunk": data_dict["image_name_chunk"][idx],
                "pairwise_predictions": (idx_res1, idx_res2),
            }

            scene_name = data_dict["scene_name"][idx]
            saving_dir = self.get_chunk_dir(scene_name)

            file_name = saving_dir / (data_dict["file_name"][idx])

            if not saving_dir.exists():
                saving_dir.mkdir(parents=True)

            torch.save(master_chunk_dict, file_name)

    @torch.no_grad()
    def prepare_data(self):
        model = Mast3rBaselineLightningModule(Mast3rBaselineConfig())
        model.eval()
        model.model.eval()
        model.model.to(get_default_device())

        batch_size = self.data_config.batch_size_mast3r

        dataloader = DataLoader(
            self.image_dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
        )

        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            total=len(self.image_dataset) // batch_size,
        ):
            self.process_chunk(batch, batch_idx, model)

        self.load_paths()
        self.prepared = True

    def load_paths(self):
        self.file_names = {}

        for scene_name in self.data_config.scenes:
            data_dir = self.get_chunk_dir(scene_name)
            if data_dir.exists():
                files = [s for s in data_dir.iterdir() if s.is_file()]
                # sorted_files = sorted(
                #     files, key=lambda f: int(str(f.name).split("_")[0])
                # )
                self.file_names[scene_name] = files

    def get_at_idx(self, idx: int):
        if self.file_names is None:
            raise ValueError(
                "No files loaded. Perhaps you forgot to call prepare_data()?"
            )

        all_files = [file for files in self.file_names.values() for file in files]
        file = all_files[idx]

        # use scene name and image name to get the corresponding occupancy and image chunk
        if not file.exists():
            print(f"File {file} does not exist. Skipping.")
        if os.path.getsize(file) < 0:
            print(f"File {file} is empty. Skipping.")

        mast3r_data = torch.load(file)

        return mast3r_data
        
