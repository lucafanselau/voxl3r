from typing import Optional
from einops import rearrange
import torch
from torch import nn
from networks.u_net import UNet3DConfig
from utils.config import BaseConfig
from jaxtyping import Float


class VoxelBasedNetworkConfig(UNet3DConfig):
    num_mlp_layers: int = 3
    dim_mlp: int = 256
    in_channels: int = 96
    

class VoxelBasedNetwork(nn.Module):
    def __init__(self, config: VoxelBasedNetworkConfig):
        super().__init__()
        self.config = config

        mlps = [nn.Conv3d(config.in_channels, config.dim_mlp, 1)]
        mlps.append(nn.ReLU())
        for _ in range(config.num_mlp_layers):
            mlps.append(nn.Conv3d(config.dim_mlp, config.dim_mlp, 1))
            mlps.append(nn.ReLU())
            
        self.mlp = nn.Sequential(*mlps)
            
        
        self.occ_predictor = nn.Conv3d(config.dim_mlp, 1, 1)
                

    def forward(
        self, x: Float[torch.Tensor, "batch channels depth height width"]
    ) -> Float[torch.Tensor, "batch 1 depth height width"]:
        
        B,P, C, X, Y, Z = x.shape
        in_enc = rearrange(x, "B P C X Y Z -> (B P) C X Y Z")
        
        dec_in = self.mlp(in_enc)
        
        occ = rearrange(self.occ_predictor(dec_in), "(B P) 1 X Y Z -> B P X Y Z", B=B, P=P)
        return occ
   
   
def main():
    pass

if __name__ == "__main__":
    main()


