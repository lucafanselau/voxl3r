from typing import List
from einops import pack, rearrange, repeat
from jaxtyping import jaxtyped, Float, Int
from beartype import beartype
from dataclasses import dataclass
import torch

from utils.config import BaseConfig

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
        dict = torch.load(path_like, weights_only=False)
        return Mast3rOutput(
            **{
                **dict,
                "res1": Mast3rResult(**dict["res1"]),
                "res2": Mast3rResult(**dict["res2"]),
            }
        )

from torch import nn

emb_dim_dec = {
    "dec_0" : 1024,
    **{"dec_{i}" : 768 for i in range(1, 13)},
}

class Mast3rModuleConfig(BaseConfig):
    model_name: str = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # possibilities ["pos", "dec_0", "dec_1", "dec_2", "dec_3", "dec_4", "dec_5", "dec_6", "dec_7", "dec_8", "dec_9", "dec_10", "dec_11", "dec_12", "res"]
    mast3r_keys: List[str] = ["dec_0"]
    
    def get_patch_emb_dim(self):
        return sum([emb_dim_dec[key] for key in self.mast3r_keys])
    
    

def load_model(model_name=None):
    from extern.mast3r.mast3r.model import AsymmetricMASt3R
    weights_path = "naver/" + model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path)
    return model

mast3r_res_keys = ["pts3d", "conf", "desc", "desc_conf"]

class Mast3rModule(nn.Module):
    def __init__(self, config: Mast3rModuleConfig):
        super().__init__()
        self.config = config
        self.model = load_model(self.config.model_name) #.half()
        self.model.eval()
        

    def forward(self, data_dict):

        self.model.eval()
        with torch.inference_mode():
            
            is_pixel = any(key in self.config.mast3r_keys for key in mast3r_res_keys)
            is_patch = any(not key in self.config.mast3r_keys for key in mast3r_res_keys)

            images = data_dict["X"]
            B, I, _2, C, H, W, = images.shape
            img1 = rearrange(images[:, :, 0, ...], "B I C H W -> (B I) C H W")
            img2 = rearrange(images[:, :, 1, ...], "B I C H W -> (B I) C H W")

            shape1 = repeat(torch.tensor(img1.shape[-2:]), "... -> B ...", B=img1.shape[0])
            shape2 = repeat(torch.tensor(img2.shape[-2:]), "... -> B ...", B=img2.shape[0])

            # compute encoder only once
            feat1, feat2, pos1, pos2 = self.model._encode_image_pairs(img1, img2, shape1, shape2)

            def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
                dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
                with torch.cuda.amp.autocast(enabled=False):
                    res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1.int())
                    res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2.int())
                return dec1, dec2, res1, res2

            # decoder 1-2
            dec1 = [feat1]
            dec2 = [feat2]
            res1 = {}
            res2 = {}
            if not ("dec_0" in self.config.mast3r_keys and self.config.mast3r_keys.__len__() == 1):
                dec1, dec2, res1, res2 = decoder(self.model, feat1, feat2, pos1, pos2, shape1, shape2)   

            # rest of the fields in a dict
            dict1 = {
                "pos1": pos1,
                **{f"dec_{i}" : dec_i for i, dec_i in enumerate(dec1)},
                **res1
            }
            
            dict2 = {
                "pos2": pos2,
                **{f"dec_{i}" : dec_i for i, dec_i in enumerate(dec2)},
                **res2,
            }


            # exclusive or
            assert is_pixel ^ is_patch, "Either patch output or pixel output must be selected (not both)"
            
            
            images_1 = rearrange(torch.cat([dict1[key] for key in self.config.mast3r_keys], dim=-1), "(B I) ... -> B I ...", B=B)
            images_2 = rearrange(torch.cat([dict2[key] for key in self.config.mast3r_keys], dim=-1), "(B I) ... -> B I ...", B=B) 

            data_dict["X"] = pack([images_1, images_2], "b i * s e")[0].detach()
            data_dict["type"] = "patch" if is_patch else "images"

        return data_dict

        
        
        
