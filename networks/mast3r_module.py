from jaxtyping import jaxtyped, Float, Int
from beartype import beartype
from dataclasses import dataclass
import torch

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
