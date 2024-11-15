import os
from typing import Tuple
import torch
from dataset import SceneDataset
from extern.mast3r.mast3r.model import AsymmetricMASt3R
from extern.mast3r.dust3r.dust3r.utils.image import load_images
from dataclasses import dataclass

from jaxtyping import jaxtyped, Float, Int
from beartype import beartype
from utils.basic import get_default_device


@jaxtyped(typechecker=beartype)
@dataclass
class Mast3rResult:
    """
    Describing normal output, based on:
    res1/res2: {
        "pts3d": torch.Size([1, 512, 384, 3])
        "conf": torch.Size([1, 512, 384])
        "desc": torch.Size([1, 512, 384, 24])
        "desc_conf": torch.Size([1, 512, 384])
    }
    typing:
    B - batch size
    W - width
    H - height
    C - #channel
    D - descriptor dimension (local features)
    """

    pts3d: Float[torch.Tensor, "B W H C"]
    conf: Float[torch.Tensor, "B W H"]
    desc: Float[torch.Tensor, "B W H D=24"]
    desc_conf: Float[torch.Tensor, "B W H"]


# Dataclass for mast3r output
@jaxtyped(typechecker=beartype)
@dataclass
class Mast3rOutput:
    """
    where 1024 is F the feature dimension
    768
    feat1: torch.Size([1, 768, 1024])
    feat2: torch.Size([1, 768, 1024])
    pos1: torch.Size([1, 768, 2])
    pos2: torch.Size([1, 768, 2])
    dec1/dec2: (list of 13 tensors) [
        torch.Size([1, 768, 1024])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
        torch.Size([1, 768, 768])
    ],

    """

    feat1: Float[torch.Tensor, "B S F"]
    feat2: Float[torch.Tensor, "B S F"]

    pos1: Int[torch.Tensor, "B S 2"]
    pos2: Int[torch.Tensor, "B S 2"]

    dec1: tuple[Float[torch.Tensor, "B S _"], ...]
    dec2: tuple[Float[torch.Tensor, "B S _"], ...]

    res1: Mast3rResult
    res2: Mast3rResult

    # reference to the original images
    img1_path: str
    img2_path: str

    # utility methods
    def save(self, path_like):
        # construct dict of self + dict for res1 and res2
        dict = {**self.__dict__, "res1": self.res1.__dict__, "res2": self.res2.__dict__}
        torch.save(dict, path_like)

    def load(path_like) -> "Mast3rOutput":
        dict = torch.load(path_like)
        return Mast3rOutput(
            **{
                **dict,
                "res1": Mast3rResult(**dict["res1"]),
                "res2": Mast3rResult(**dict["res2"]),
            }
        )


# Taken from `sparse_ga.py` inside of mast3r
@torch.no_grad()
def inference(
    model: AsymmetricMASt3R, img1, img2, img1_path, img2_path, device
) -> Mast3rOutput:
    shape1 = torch.from_numpy(img1["true_shape"]).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2["true_shape"]).to(device, non_blocking=True)
    img1 = img1["img"].to(device, non_blocking=True)
    img2 = img2["img"].to(device, non_blocking=True)

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
    # decoder 2-1
    # res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return Mast3rOutput(
        feat1=feat1,
        feat2=feat2,
        pos1=pos1,
        pos2=pos2,
        dec1=dec1,
        dec2=dec2,
        # Convert result dict to Mast3rResult
        res1=Mast3rResult(**res1),
        res2=Mast3rResult(**res2),
        img1_path=img1_path,
        img2_path=img2_path,
    )


def load_model(model_name=None, device=get_default_device()):
    weights_path = "naver/" + model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def predict(model, folder):
    # this is also the default for the demo
    image_size = 512
    image_names = sorted(os.listdir(folder))
    images = load_images(folder, image_size)

    # naive way to make pairs out of the image, by grouping (1, 2), (3, 4), ...
    pairs_in = [([images[i], images[i + 1]]) for i in range(0, len(images), 2)]

    predictions = inference(
        model,
        pairs_in[0][0],
        pairs_in[0][1],
        image_names[0],
        image_names[1],
        device=get_default_device(),
    )

    return predictions


def main():

    args = {
        "local_network": False,
        "server_name": None,
        "image_size": 512,
        "server_port": None,
        "weights": None,
        "model_name": "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "device": "mps",
        "tmp_dir": None,
        "silent": False,
        "share": False,
        "gradio_delete_cache": None,
    }

    dataset = SceneDataset(
        data_dir="/home/luca/mnt/data/scannetpp/data",
        camera="iphone",
        n_points=300000,
        threshold_occ=0.01,
        representation="occ",
        visualize=True,
        max_seq_len=max_seq_len,
        resolution=0.02,
    )

    model = load_model(args["model_name"])
    predictions = predict(model, "data/test_images")
    predictions.save("predictions.pt")


if __name__ == "__main__":
    main()
