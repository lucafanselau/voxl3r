from einops import rearrange
from torch import nn
from positional_encodings.torch_encodings import PositionalEncoding3D


class PositionalEncoding(nn.Module):
    """
    Transform that smears images with known camera parameters and N channels
    into a feature grid via projection of unknown depth and trilinear interpolation
    """

    def __init__(
        self,
        *_args
    ):
        super().__init__()
        self.pe = None
        
    def __call__(self, data: dict) -> dict:

        data["X"] = rearrange(data["X"], "P I C X Y Z -> P X Y Z (I C)")   
         
        if self.pe is None:
            self.pe = PositionalEncoding3D(data["X"].shape[-1]).to(data["X"].device)

        pe_tensor = self.pe(data["X"])
        data["X"] = data["X"] + pe_tensor
        
        data["X"] = rearrange(data["X"], "... X Y Z C -> ... C X Y Z")
        
        return data