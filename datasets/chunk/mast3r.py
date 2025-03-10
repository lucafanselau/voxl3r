from dataclasses import dataclass
import os
from pathlib import Path
from typing import Generator, Optional, Tuple
from loguru import logger
from typing_extensions import TypedDict

from einops import rearrange
import torch
from torch.utils.data import DataLoader
from jaxtyping import Float
from datasets.chunk.base import ChunkBaseDataset, ChunkBaseDatasetConfig
from datasets.chunk.pair_matching import PairMatching
from datasets.chunk import image
from utils.basic import get_default_device
import numpy as np
from tqdm import tqdm
import shutil


from datasets import scene_processed
import gc

from utils.data_parsing import get_camera_params


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
    mast3r_data_dir: Optional[str | list[str]] = (
        None  # fallback to data_dir if not provided
    )
    folder_name_mast3r: Optional[str] = "preprocessed_pairs"
    batch_size_mast3r: int
    force_prepare_mast3r: bool
    # pair_matching: str = "first_centered"
    mast3r_keys: Optional[list[str]] = (
        None  # valid keys are pts3d, conf, desc, desc_conf, feat, pos, dec_0, dec_1, ... , dec_12
    )


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
    pairs_image_names: list[Tuple[str, str]]


class Dataset(ChunkBaseDataset):
    def __init__(
        self,
        data_config: Config,
        base_dataset: scene_processed.Dataset,
        image_dataset: image.Dataset,
    ):
        super(Dataset, self).__init__(data_config, base_dataset)
        self.image_dataset = image_dataset
        self.data_config = data_config
        # self.pair_matching = PairMatching[data_config.pair_matching]()

    # THIS NOW EXPECTS A BATCH
    def load_prepare(self, item):
        data_dict = item
        image_names = data_dict["image_pairs"]
        camera_dir = [
            self.base_dataset.get_image_dir(scene_name)
            for scene_name in item["scene_name"]
        ]
        image_pairs = {}
        for j in range(self.data_config.batch_size_mast3r):
            image_pairs[f"{item['scene_name'][j]}_{item['identifier'][j]}"] = [
                (
                    camera_dir[j] / image_names[i][0][j],
                    camera_dir[j] / image_names[i][1][j],
                )
                for i in range(self.data_config.num_pairs)
            ]

        return image_pairs

    def load_images(self, image_pairs, size=512):
        from extern.mast3r.dust3r.dust3r.utils.image import load_images

        mast3r_dicts = {}

        for identifier in image_pairs.keys():
            mast3r_dicts[identifier] = load_images(
                [str(ele) for pair in image_pairs[identifier] for ele in pair],
                size,
                verbose=False,
            )

        return mast3r_dicts

    def get_saving_path(self, scene_name: str) -> Path:
        if isinstance(self.data_config.mast3r_data_dir, list):
            # check if at any of the data_dirs the scene_name exists
            for data_dir in self.data_config.mast3r_data_dir:
                if (
                    Path(data_dir)
                    / self.data_config.storage_preprocessing
                    / scene_name
                    / self.data_config.camera
                ).exists():
                    return (
                        Path(data_dir)
                        / self.data_config.storage_preprocessing
                        / scene_name
                        / self.data_config.camera
                    )
            # otherwise find the dir with the most available space
            best_dir = max(
                self.data_config.mast3r_data_dir,
                key=lambda x: shutil.disk_usage(x).free,
            )
            return (
                Path(best_dir)
                / self.data_config.storage_preprocessing
                / scene_name
                / self.data_config.camera
            )
        else:
            data_dir = self.data_config.mast3r_data_dir or self.data_config.data_dir
            return (
                Path(data_dir)
                / self.data_config.storage_preprocessing
                / scene_name
                / self.data_config.camera
            )

    def get_chunk_dir(self, scene_name):

        dc = self.data_config
        base_folder_name = f"num_pairs_{dc.num_pairs}_chunk_extent_{dc.chunk_extent}"

        path = (
            self.get_saving_path(scene_name)
            / self.data_config.folder_name_mast3r
            / base_folder_name
        )

        return path

    def check_chunks_exist(self, batch) -> bool:
        """Check if all chunks in batch already exist"""
        B = len(batch["identifier"])
        return all(
            self.get_chunk_dir(batch["scene_name"][idx])
            .joinpath(batch["file_name"][idx])
            .exists()
            for idx in range(B)
        )

    @torch.no_grad()
    def process_chunk(
        self, batch, batch_idx, model
    ) -> Generator[tuple[Output, Path, str], None, None]:
        try:
            data_dict = batch

            image_paths = self.load_prepare(batch)

            if not self.data_config.force_prepare_mast3r and self.check_chunks_exist(
                batch
            ):
                return

            mast3r_dict = self.load_images(image_paths)

            # images is B, num_pairs*2, 1, 3, H, W
            images = torch.stack(
                [
                    torch.stack(
                        [
                            mast3r_dict[key][i]["img"]
                            for i in range(self.data_config.num_pairs * 2)
                        ]
                    )
                    for key in mast3r_dict.keys()
                ]
            ).to(get_default_device())
            true_shapes = torch.from_numpy(
                np.stack(
                    [
                        np.stack(
                            [
                                mast3r_dict[key][i]["true_shape"]
                                for i in range(self.data_config.num_pairs * 2)
                            ]
                        )
                        for key in mast3r_dict.keys()
                    ]
                )
            ).to(get_default_device())

            res1, res2, dict1, dict2 = model.forward(
                rearrange(
                    images[:, ::2, ...], "B SEQ_LEN 1 C H W -> (B SEQ_LEN) C H W"
                ),
                rearrange(
                    images[:, 1::2, ...], "B SEQ_LEN 1 C H W -> (B SEQ_LEN) C H W"
                ),
                rearrange(true_shapes[:, ::2, ...], "B SEQ_LEN 1 C -> (B SEQ_LEN) C"),
                rearrange(true_shapes[:, 1::2, ...], "B SEQ_LEN 1 C -> (B SEQ_LEN) C"),
            )

            combined_dicts1 = {**res1, **dict1}
            combined_dicts2 = {**res2, **dict2}
            res1 = (
                {key: combined_dicts1[key] for key in self.data_config.mast3r_keys}
                if self.data_config.mast3r_keys is not None
                else combined_dicts1
            )
            res2 = (
                {key: combined_dicts2[key] for key in self.data_config.mast3r_keys}
                if self.data_config.mast3r_keys is not None
                else combined_dicts2
            )

            B = mast3r_dict.keys().__len__()

            for i, identifier in enumerate(mast3r_dict.keys()):

                idx_res1 = {
                    key: rearrange(
                        res1[key],
                        "(B SEQ_LEN) ... -> B SEQ_LEN ...",
                        SEQ_LEN=self.data_config.num_pairs,
                        B=B,
                    )[i, ...]
                    .detach()
                    .cpu()
                    for key in res1.keys()
                }

                idx_res2 = {
                    key: rearrange(
                        res2[key],
                        "(B SEQ_LEN) ... -> B SEQ_LEN ...",
                        SEQ_LEN=self.data_config.num_pairs,
                        B=B,
                    )[i, ...]
                    .detach()
                    .cpu()
                    for key in res2.keys()
                }

                # i = data_dict["identifier"].index(identifier)

                scene_name = data_dict["scene_name"][i]
                scene_path = self.base_dataset.data_dir / scene_name
                images_to_load = [
                    image_path.name
                    for image_pairs in image_paths[identifier]
                    for image_path in image_pairs
                ]
                images_with_params = get_camera_params(
                    scene_path, self.base_dataset.camera, images_to_load, 0
                )

                master_chunk_dict = {
                    "scene_name": scene_name,
                    "file_name": data_dict["file_name"][i],
                    "identifier": data_dict["identifier"][i],
                    "pairwise_predictions": (idx_res1, idx_res2),
                    "pairs_image_paths": image_paths[identifier],
                    "camera_params": images_with_params,
                }

                saving_dir = self.get_chunk_dir(scene_name)

                if not saving_dir.exists():
                    saving_dir.mkdir(parents=True)

                yield master_chunk_dict, saving_dir, data_dict["identifier"][i]

                # Clean up individual item resources
                del master_chunk_dict
                gc.collect()

        finally:
            # Ensure cleanup of batch resources
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def prepare_data(self):
        if self.data_config.skip_prepare and not self.data_config.force_prepare_mast3r:
            self.load_paths()
            self.on_after_prepare()
            self.prepared = True
            return

        # NOTE: we should probably not import from experiments here
        from experiments.mast3r_baseline.module import (
            Mast3rBaselineConfig,
            Mast3rBaselineLightningModule,
        )

        model = Mast3rBaselineLightningModule(Mast3rBaselineConfig())
        model.eval()
        model.model.eval()
        model.model.to(get_default_device())

        batch_size = self.data_config.batch_size_mast3r

        dataloader = DataLoader(
            self.image_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )

        # error_messages = []
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            total=len(self.image_dataset) // batch_size,
        ):
            try:
                for master_chunk_dict, saving_dir, file_name in self.process_chunk(
                    batch, batch_idx, model
                ):
                    torch.save(master_chunk_dict, saving_dir / file_name)
            except Exception as e:
                logger.error(f"Error processing chunk {batch_idx}: {e}")
                # error_messages.append(f"Error processing chunk {batch_idx}: {e}")

        # for message in error_messages:
        #     print(message)

        self.load_paths()
        self.on_after_prepare()
        self.prepared = True

    def load_paths(self):
        self.file_names = {}

        if self.data_config.split is not None:
            iterator = self.base_dataset.scenes
        else:
            iterator = (
                self.data_config.scenes
                if self.data_config.scenes is not None
                else self.base_dataset.scenes
            )

        for scene_name in iterator:
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

        try:
            mast3r_data = torch.load(file, weights_only=False)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            return self.get_at_idx(idx - 1)

        return mast3r_data
