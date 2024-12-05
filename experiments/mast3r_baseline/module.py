from dataclasses import dataclass

from einops import rearrange
import lightning.pytorch as pl
import torch
import open3d as o3d
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)


from experiments.surface_net_3d.visualize import (
    VoxelGridVisualizer,
    VoxelVisualizerConfig,
)
from extern.mast3r.mast3r.model import AsymmetricMASt3R
from utils.transformations import invert_pose, invert_pose_batched
from utils.visualize import visualize_mesh


@dataclass
class Mast3rBaselineConfig:
    model_name: str = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def load_model(model_name=None):
    weights_path = "naver/" + model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path)
    return model


class Mast3rBaselineLightningModule(pl.LightningModule):
    def __init__(self, config: Mast3rBaselineConfig):
        super().__init__()
        self.config = config

        self.model = load_model(self.config.model_name)
        self.model.eval()

        metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
                "auroc": BinaryAUROC(),
            }
        )

        self.test_metrics = metrics.clone(prefix="test_")

    def _shared_step(self, batch, batch_idx):
        code = 0
        pass

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, img1, img2, shape1, shape2):
        model = self.model
        # compute encoder only once
        feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

        def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
            dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
            with torch.cuda.amp.autocast(enabled=False):
                res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
            return dec1, dec2, res1, res2

        # decoder 1-2
        dec1, dec2, res1, res2 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)

        # rest of the fields in a dict
        dict1 = {
            "feat": feat1,
            "pos": pos1,
            "dec": dec1,
        }
        dict2 = {
            "feat": feat2,
            "pos": pos2,
            "dec": dec2,
        }

        return res1, res2, dict1, dict2

    def test_step(self, batch, batch_idx):
        occ, images, image_names, transforms, data_dict = batch

        # collate gives us something like this:
        # images: list[{ "img": FloatTensor[batch_size 1 3 336 512], "true_shape": IntTensor[batch_size 1 2] }] with len: seq_len
        # image_names: list[array[str, batch_size]] with len: seq_len
        # transforms: list[{
        #     "R_cw": FloatTensor[batch_size 3 3],
        #     "t_cw": FloatTensor[batch_size 3 1],
        #     "T_cw": FloatTensor[batch_size 4 4], -> Homogeneous transformation matrix
        #     "K": FloatTensor[batch_size 3 3], -> Intrinsic camera matrix
        #     "dist_coeffs": FloatTensor[batch_size 4], -> Distortion coefficients -> (Always 0?)
        #     "width": FloatTensor[batch_size], -> Image width
        #     "height": FloatTensor[batch_size], -> Image height
        # }] with len: seq_len

        # go over images in pairs
        pc = []

        def process_pair(i):
            img_dict1, img_dict2 = images[i], images[i + 1]
            img1, img2 = img_dict1["img"].squeeze(1), img_dict2["img"].squeeze(1)
            true_shape1, true_shape2 = img_dict1["true_shape"].squeeze(1), img_dict2[
                "true_shape"
            ].squeeze(1)

            res1, res2 = self.forward(img1, img2, true_shape1, true_shape2)
            transform1, transform2 = transforms[i], transforms[i + 1]

            T_wc = invert_pose_batched(transform1["R_cw"], transform1["t_cw"])

            # now we need to transform the points is shape torch.Size([16, 336, 512, 3])
            points = torch.stack([res["pts3d"] for res in [res1, res2]], dim=1)

            # extend points to be homogeneous vector
            points = torch.cat(
                [points, torch.ones((*points.shape[:-1], 1)).to(points)], dim=-1
            )
            points = rearrange(points, "B TWO W H C -> B (TWO W H) C 1")
            T_wc = rearrange(T_wc, "B F D -> B 1 F D").to(points)
            # Transformation from world to camera frame 1
            T_0w = transforms[0]["T_cw"]
            T_0w = rearrange(T_0w, "B F D -> B 1 F D").to(points)

            points_0 = torch.matmul(T_0w, torch.matmul(T_wc, points))
            points_0 = points_0.squeeze(-1)

            return points_0  # [B, #P, 4]

        results = torch.stack(
            [process_pair(i) for i in range(0, len(images), 2)], dim=1
        )

        points = rearrange(results, "batch P N C -> batch (P N) C")[:, :, :3]
        center = data_dict["center"].unsqueeze(-2)
        centered_points = points - center
        res = data_dict["resolution"].unsqueeze(-1)
        extents = (data_dict["grid_size"] * res).unsqueeze(-2)
        min_bound = -extents / 2
        max_bound = extents / 2
        # bounded_points = points[(points >= min_bound).any(axis=1) & (points < max_bound).any(axis=1)]
        fractional_points = (centered_points - min_bound) / extents

        voxel_points = fractional_points * data_dict["grid_size"].unsqueeze(-2)
        voxels_indices = voxel_points.int()

        voxel_grid = (
            torch.zeros(
                (data_dict["grid_size"].shape[0], 1, *data_dict["grid_size"][0])
            )
            .to(points)
            .int()
        )
        B = voxel_grid.shape[0]
        for i in range(B):
            indices_mask = (
                (voxels_indices[i] >= data_dict["grid_size"][i] * 0)
                & (voxels_indices[i] < data_dict["grid_size"][i])
            ).all(axis=-1)
            masked_indices = voxels_indices[i][indices_mask].long()
            voxel_grid[
                i, 0, masked_indices[:, 0], masked_indices[:, 1], masked_indices[:, 2]
            ] = 1

        metrics = self.test_metrics(voxel_grid, occ.int())

        def visualize_voxel_grid():
            # visualize a single prediction
            visualizer = VoxelGridVisualizer(VoxelVisualizerConfig())

            grid = torch.stack([voxel_grid[0], occ[0]], dim=0)

            visualizer.visualize_batch(
                torch.ones_like(grid).repeat(1, 3, 1, 1, 1) * 212,
                grid.squeeze(1),
                labels=["Prediction", "Occupancy"],
            )

        self.log_dict(metrics, prog_bar=True, on_step=True)

        return
