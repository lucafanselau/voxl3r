from einops import rearrange
import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
import lightning as pl
from positional_encodings.torch_encodings import PositionalEncoding3D, get_emb
from torch import nn
from dataset import SceneDataset, SceneDatasetTransformToTorch
from torch.utils.data import Dataset, Subset


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
        self, base_dataset: SceneDataset, scene_id=None, p_batch_size=1024, channels=0
    ):
        super().__init__()
        self.dataset = base_dataset
        self.scene_id = scene_id
        self.p_batch_size = p_batch_size
        self.channels = channels

    def prepare_data(self):
        self.chunks = self.dataset.chunk_whole_scene(self.scene_id)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):

        idx = 8

        images, transformations, points, gt = SceneDatasetTransformToTorch(
            "cuda"
        ).forward(self.chunks[idx])
        images = images / 255.0

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
        # Initialize a new tensor of shape [128, 30], filled with -1s
        X_extended = -1 * torch.ones(X.shape[0], X.shape[1]).to(X.device)

        # Copy the original tensor into the new tensor
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
        batch_size=512,
        max_sequence_length=20,
        target_device="mps",
        single_chunk=True,
        channels=24,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.single_chunk = single_chunk
        self.transform = SceneDatasetTransformToTorch(target_device)
        self.dataset = dataset
        self.channels = channels
        if single_chunk:
            self.scene_id = scene_id
            self.scene_idx = dataset.get_index_from_scene(scene_id)
            self.max_sequence_length = max_sequence_length
            self.parsed = None
        else:
            if isinstance(scene_id, list) and len(scene_id) == 3:
                self.scene_idx_train, self.scene_idx_val, self.scene_idx_test = (
                    dataset.get_index_from_scene(scene_id)
                )
                self.dataset = OccSurfaceNetDataset(
                    dataset, self.scene_idx_train, p_batch_size=None, channels=channels
                )
                # here functionality to use multple different scenes for train, val and test can be added
            else:
                raise ValueError(
                    "If single_chunk is False, scene_id should be a list of length 3"
                )

    def prepare_data(self):
        if self.single_chunk:
            scene_dataset = self.dataset
            idx = scene_dataset.get_index_from_scene(self.scene_id)
            data = scene_dataset[idx]
            mesh = data["mesh"]
            points, gt = data["training_data"]
            image_names, camera_params_list, _ = data["images"]

            images, transformations, points, gt = self.transform.forward(data)
            # and normalize images
            images = images / 255.0

            X = project_points_to_images(points, images, transformations)

            # we need to pad X to contain 3 * self.max_sequence_length elements in the last dimension (fill_value is -1)
            fill_value = -1.0
            num_points, num_images_flattened = X.shape
            X = torch.nn.functional.pad(
                X,
                (0, 3 * self.max_sequence_length - num_images_flattened),
                value=fill_value,
            )

            # target values will be gt
            target = rearrange(gt, "points -> points 1")

            # Dataloader for [X, target] Tensor
            # split the data
            dataset = torch.utils.data.TensorDataset(X, target, points)

            return dataset
        else:
            self.dataset.prepare_data()

    def setup(self, stage):
        if self.single_chunk:
            if self.parsed is None:
                # self.parsed = OccSurfaceNetDataset(self.dataset)
                self.parsed = self.prepare_data()
                generator = torch.Generator().manual_seed(42)
                self.split = torch.utils.data.random_split(
                    self.parsed, [0.6, 0.2, 0.2], generator=generator
                )

            if stage == "fit" or stage is None:
                self.train_dataset = self.split[0]
                self.val_dataset = self.split[1]

            if stage == "test" or stage is None:
                self.test_dataset = self.split[2]

        else:

            generator = torch.Generator().manual_seed(42)
            self.split = torch.utils.data.random_split(
                self.dataset, [1.0, 0.0, 0.0], generator=generator
            )

            if stage == "fit" or stage is None:
                self.train_dataset = self.split[0]
                self.val_dataset = self.split[0]

            if stage == "test" or stage is None:
                self.test_dataset = self.split[0]

    def train_dataloader(self):
        if self.single_chunk:
            return torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
            )

    def val_dataloader(self):
        if self.single_chunk:
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
            )

    def test_dataloader(self):
        if self.single_chunk:
            return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            return torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=custom_collate_fn,
            )

    # def predict_dataloader(self):
    #   return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
