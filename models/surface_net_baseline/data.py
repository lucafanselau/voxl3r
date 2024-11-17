from einops import rearrange
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
import lightning as pl
from torch import nn
from dataset import SceneDataset, SceneDatasetTransformToTorch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from utils.chunking import create_chunk, mesh_2_voxels


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

    def forward(self, tensor):
        """
        :param tensor: A 2d tensor of size (batch_size, 3)
        :return: Positional Encoding Matrix of size (batch_size, self.channels * 3)
        """

        # if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
        #     return self.cached_penc

        # self.cached_penc = None
        # batch_size, x, y, z, orig_ch = tensor.shape
        # pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        # pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        # pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_x = tensor[:, 0]
        pos_y = tensor[:, 1]
        pos_z = tensor[:, 2]
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb_y = get_emb(sin_inp_y)
        emb_z = get_emb(sin_inp_z)
        # emb = torch.zeros(
        #     (x, y, z, self.channels * 3),
        #     device=tensor.device,
        #     dtype=(
        #         self.dtype_override if self.dtype_override is not None else tensor.dtype
        #     ),
        # )
        # emb[:, :, :, : self.channels] = emb_x
        # emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        # emb[:, :, :, 2 * self.channels :] = emb_z
        emb = torch.cat([emb_x, emb_y, emb_z], dim=1)

        # self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb


def get_3d_pe(points, channels):
    pe = PositionalEncoding3D(channels).to(points.device)
    return pe(points)


@jaxtyped(typechecker=beartype)
def project_points_to_images(
    points: Float[Tensor, "Points 3"],
    images: Float[Tensor, "Images 3 H W"],
    transformations: Float[Tensor, "Images 3 4"],
    add_positional_encoding: bool = False,
    channels: int = 12,
) -> Float[Tensor, "Points I"]:
    """
    I = Images * 3 + channels
    images: expected to be normalized [0, 1] float images C, H, W

    """
    # reshape points [num_points, 3] so that we have [num_points, num_images, 3]
    num_points = points.shape[0]
    num_images, _3, height, width = images.shape

    pe = None
    if add_positional_encoding:
        pe = get_3d_pe(points, channels)

    points = points.reshape(num_points, 1, 3).repeat(1, num_images, 1)
    # convert to homographic coordinates
    points = torch.cat(
        [points, torch.full((num_points, num_images, 1), 1).to(points)], dim=-1
    )
    # and convert it to "matrix" (4x1) (num_points, num_images, 4, 1)
    points = rearrange(points, "n i d -> n i d 1")

    # perform the matmul with broadcasting (num_points, num_images, 4, 4) x (num_points, num_images, 4, 1) -> (num_points, num_images, 3, 1)
    transformed_points = torch.matmul(transformations, points)

    # remove last dimension (num_points, num_images, 3)
    transformed_points = rearrange(transformed_points, "n i d 1 -> n i d")

    # perform perspective division
    transformed_points = transformed_points[:, :, :3] / transformed_points[
        :, :, 2
    ].unsqueeze(-1)
    # and drop the last dimension
    transformed_points = transformed_points[:, :, :2]

    # now we need to mask by the bounding box of the image and replace with a constant fill (-inf)
    fill_value = -1.0
    mask = (
        (transformed_points[:, :, 0] >= 0)
        & (transformed_points[:, :, 0] < width)
        & (transformed_points[:, :, 1] >= 0)
        & (transformed_points[:, :, 1] < height)
    )
    full_mask = mask.reshape(-1, num_images, 1).repeat(1, 1, 2)
    valid_pixels = torch.where(full_mask, transformed_points, torch.nan)

    # now sample the image at the valid pixels
    # the way this is done currently all of the "invalid" uvs will be 0, but we only use the valid ones later so its fine
    grid = rearrange(valid_pixels, "p n two -> n 1 p two", two=2)
    # normalize in last dimension to range [-1, 1]
    grid = (grid / torch.tensor([width, height]).to(grid)) * 2 - 1

    sampled = torch.nn.functional.grid_sample(images, grid)
    sampled = rearrange(sampled, "images channels 1 points -> points images channels")

    sampled[~mask] = fill_value

    # reshape to (num_points, num_images * 3)
    sampled = rearrange(sampled, "points images channels -> points (images channels)")

    if add_positional_encoding:
        sampled = torch.cat([sampled, pe], dim=-1)

    return sampled


class OccSurfaceNetDataset(Dataset):
    def __init__(
        self,
        base_dataset: SceneDataset,
        scene_name=None,
        p_batch_size=1024,
        channels=0,
        max_seq_len=8,
        image_name=None,
        even_distribution=True,
        len_chunks=None,
    ):
        super().__init__()
        self.dataset = base_dataset
        self.scene_name = scene_name
        self.scene_idx = base_dataset.get_index_from_scene(scene_name)

        data = self.dataset[self.scene_idx]
        self.mesh = data["mesh"]
        self.path_images = data["path_images"]
        self.camera_params = data["camera_params"]
        self.p_batch_size = p_batch_size
        self.max_seq_len = max_seq_len
        self.channels = channels
        self.image_name = image_name
        self.even_distribution = even_distribution
        self.len_chunks = len_chunks

    def prepare_data(self, even_distribution=False):
        self.chunks = []
        image_names = list(self.camera_params.keys())

        if self.image_name is not None:
            image_names = len(self.image_name) * [self.image_name]

        print("Preparing chunks for training:")
        for i in tqdm(range((len(image_names) // self.max_seq_len))):
            if self.len_chunks is not None and i == self.len_chunks:
                break
            image_name = image_names[i * self.max_seq_len]
            data_chunk = create_chunk(
                self.mesh.copy(),
                image_name,
                self.camera_params,
                max_seq_len=self.max_seq_len,
                image_path=self.path_images,
            )
            voxel_grid, coordinates, occupancy_values = mesh_2_voxels(
                data_chunk["mesh"]
            )

            trainings_dict = {
                "training_data": (coordinates, occupancy_values),
                "images": (
                    data_chunk["image_names"],
                    data_chunk["camera_params"].values(),
                ),
            }

            self.chunks.append(trainings_dict)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):

        images, transformations, points, gt = SceneDatasetTransformToTorch(
            "cuda"
        ).forward(self.chunks[idx])
        images = images / 255.0

        if self.even_distribution:
            false_indices = torch.where(gt == 0.0)[0]
            true_indices = torch.where(gt == 1.0)[0]

            false_indices_perm = torch.randperm(true_indices.size(0))
            false_indices = false_indices[false_indices_perm]
            gt = torch.cat([gt[true_indices], gt[false_indices]])
            points = torch.cat([points[true_indices], points[false_indices]])

        if self.p_batch_size is not None:
            indices = np.random.choice(points.shape[0], self.p_batch_size, replace=True)
            points = points[indices]
            gt = gt[indices]

        X = project_points_to_images(
            points,
            images,
            transformations,
            add_positional_encoding=True if self.channels != 0 else False,
            channels=self.channels,
        )

        # makes sure all vector have the same second dimension
        X_extended = -1 * torch.ones(X.shape[0], 3 * self.max_seq_len).to(X.device)
        X_extended[:, : X.shape[1]] = X

        return X_extended, gt.unsqueeze(1), points


def custom_collate_fn(batch):
    batch_stacked = torch.cat([torch.cat(mini_batch, axis=1) for mini_batch in batch])
    return (
        batch_stacked[:, :-4],
        batch_stacked[:, -4].unsqueeze(1),
        batch_stacked[:, -3:],
    )


class OccSurfaceNetDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: SceneDataset,
        scene_id: str,
        batch_size=1,
        p_in_batch=2048,
        max_seq_len=20,
        target_device="mps",
        channels=24,
        image_name=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = SceneDatasetTransformToTorch(target_device)
        self.dataset = dataset
        self.channels = channels
        self.image_name = image_name

        self.dataset = OccSurfaceNetDataset(
            dataset,
            scene_id,
            p_batch_size=p_in_batch,
            channels=channels,
            max_seq_len=max_seq_len,
            image_name=image_name,
        )

    def prepare_data(self):
        self.dataset.prepare_data(even_distribution=True)

    def setup(self, stage):

        generator = torch.Generator().manual_seed(42)

        if self.image_name is None:
            self.split = torch.utils.data.random_split(
                self.dataset, [0.6, 0.2, 0.2], generator=generator
            )

            if stage == "fit" or stage is None:
                self.train_dataset = self.split[0]
                self.val_dataset = self.split[1]

            if stage == "test" or stage is None:
                self.test_dataset = self.split[2]
        else:
            self.split = torch.utils.data.random_split(
                self.dataset, [1.0, 0.0, 0.0], generator=generator
            )

            if stage == "fit" or stage is None:
                self.train_dataset = self.split[0]
                self.val_dataset = self.split[0]

            if stage == "test" or stage is None:
                self.test_dataset = self.split[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    # def predict_dataloader(self):
    #   return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
