from dataclasses import dataclass

import lightning.pytorch as pl
import torch

from extern.mast3r.mast3r.model import AsymmetricMASt3R


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

        return res1, res2

    def test_step(self, batch, batch_idx):
        # FIXME: collate fn currently creates bs here
        images, image_names, transforms = batch

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
        for i in range(0, len(images), 2):
            img_dict1, img_dict2 = images[i], images[i + 1]
            img1, img2 = img_dict1["img"].squeeze(1), img_dict2["img"].squeeze(1)
            true_shape1, true_shape2 = img_dict1["true_shape"].squeeze(1), img_dict2[
                "true_shape"
            ].squeeze(1)

            res1, res2 = self.forward(img1, img2, true_shape1, true_shape2)
            transform1, transform2 = transforms[i], transforms[i + 1]

        return self._shared_step(batch, batch_idx)
