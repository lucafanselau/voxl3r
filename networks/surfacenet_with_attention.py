from typing import Optional
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.attention_net import AttentionNetConfig
from networks.attention_net import Transformer
from networks.surfacenet import SurfaceNet


class BaseSurfaceNetConfig(AttentionNetConfig):
    """Base configuration for SurfaceNet architecture"""

    # Input configuration
    in_channels: int  # local features
    upscale_side_dim: bool = False
    requires_grad: bool = False
    
    surfacenet_path: Optional[str] = "/home/luca/uni/master/dl-in-vc/voxl3r/.lightning/mast3r-3d-experiments/mast3r-3d-experiments/uwxc72cp/checkpoints/epoch=33-step=10098-val_loss=0.63 copy.ckpt"


class SmallSurfaceNetConfig(BaseSurfaceNetConfig):
    """Small SurfaceNet configuration - original dimensions from the paper"""

    # Layer dimensions (output channels)
    l1_dim: int = 32  # Layer 1 output dimension
    l2_dim: int = 80  # Layer 2 output dimension
    l3_dim: int = 160  # Layer 3 output dimension
    l4_dim: int = 300  # Layer 4 output dimension (dilated)
    l5_dim: int = 100  # Layer 5 output dimension (aggregator)

    # Side output dimension
    side_dim: int = 16  # Dimension for all side outputs


class MediumSurfaceNetConfig(BaseSurfaceNetConfig):
    """Medium SurfaceNet configuration - 1.5x increased capacity"""

    # Layer dimensions (output channels)
    l1_dim: int = 48  # Layer 1 output dimension (1.5x)
    l2_dim: int = 128  # Layer 2 output dimension (1.5x)
    l3_dim: int = 256  # Layer 3 output dimension (1.5x)
    l4_dim: int = 480  # Layer 4 output dimension (1.5x)
    l5_dim: int = 160  # Layer 5 output dimension (1.5x)

    # Side output dimension
    side_dim: int = 24  # Dimension for all side outputs (1.5x)


class LargeSurfaceNetConfig(BaseSurfaceNetConfig):
    """Large SurfaceNet configuration - 2x increased capacity"""

    # Layer dimensions (output channels)
    l1_dim: int = 64  # Layer 1 output dimension (2x)
    l2_dim: int = 192  # Layer 2 output dimension (2x)
    l3_dim: int = 384  # Layer 3 output dimension (2x)
    l4_dim: int = 512  # Layer 4 output dimension (2x)
    l5_dim: int = 256  # Layer 5 output dimension (2x)

    # Side output dimension
    side_dim: int = 32  # Dimension for all side outputs (2x)


def make_conv_block(in_ch, out_ch, kernel_size=3, dilation=1, padding=None):
    """
    Creates a 3D convolution -> batchnorm -> ReLU block.
    """
    if padding is None:
        padding = dilation  # a typical choice so that spatial size stays the same
    return nn.Sequential(
        nn.Conv3d(
            in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False
        ),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


class SurfaceNetTransformer(nn.Module):
    """
    A PyTorch module implementing the core 3D SurfaceNet architecture
    based on Table 1 and Figure 3 of the paper.
    """

    def __init__(self, config):
        """
        in_channels = 48: local features
        """
        super(SurfaceNetTransformer, self).__init__()
        if config.surfacenet_path is not None and "upscale_side_dim" in config.model_fields_set and config.upscale_side_dim:
            config.side_dim = 32
        self.config = config

        self.model_surfacenet = SurfaceNet(config)
        # load surfacenet and freeze weights
        
        if config.surfacenet_path is not None:
            loaded_ckpt = torch.load(config.surfacenet_path, weights_only=False)
            state_dict = {".".join(k.split(".")[2:]) : v for k, v in loaded_ckpt["state_dict"].items() if k.split(".")[1] == "0"}

            self.model_surfacenet.load_state_dict(state_dict)
            for param in self.model_surfacenet.parameters():
                param.requires_grad = config.requires_grad if "requires_grad" in config.model_fields_set else False
            for param in self.model_surfacenet.l5.parameters():
                param.requires_grad = True  
            for param in self.model_surfacenet.out_conv.parameters():
                param.requires_grad = True  
            
        # add attention mechanism to let pairs communicate
        X, Y, Z = config.grid_size_sample
        p = config.num_pairs
        self.instance_embedding = nn.Embedding(config.num_pairs, config.dim)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('(b p) ic x y z -> (b x y z) p ic', p=p), # batch+pair images+channels x y z
            nn.LayerNorm(config.side_dim*4),
            nn.Linear(config.side_dim * 4, config.dim),
            nn.LayerNorm(config.dim),
        )
        self.transformer = Transformer(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, seq_len=p, no_attn_feedthrough=False, num_pairs=config.num_pairs)
        self.to_conv_embedding = nn.Sequential(
            Rearrange('(b x y z) p ic -> (b p) ic x y z', x=X, y=Y, z=Z),
            nn.InstanceNorm3d(config.dim),
            nn.Conv3d(config.dim, config.side_dim*4, kernel_size=1, bias=False),
            nn.InstanceNorm3d(config.side_dim*4),
        )

    def forward(self, batch):
        """
        Forward pass through the network.
        """
        x = batch["X"]
        B, P, I, C, X, Y, Z = x.shape
        x = rearrange(x, "B P I C X Y Z -> (B P) (I C) X Y Z")

        # --- l1
        x1 = self.model_surfacenet.l1(x)

        # side 1
        s1 = self.model_surfacenet.s1(x1)
        s1 = torch.sigmoid(s1)  # side outputs use sigmoid

        # pooling
        p1 = self.model_surfacenet.p1(x1)

        # --- l2
        x2 = self.model_surfacenet.l2(p1)

        # side 2
        s2 = self.model_surfacenet.s2(x2)
        s2 = torch.sigmoid(s2)

        # pooling
        p2 = self.model_surfacenet.p2(x2)

        # --- l3
        x3 = self.model_surfacenet.l3(p2)

        # side 3
        s3 = self.model_surfacenet.s3(x3)
        s3 = torch.sigmoid(s3)

        # --- l4 (dilated)
        x4 = self.model_surfacenet.l4(x3)

        # side 4
        s4 = self.model_surfacenet.s4(x4)
        s4 = torch.sigmoid(s4)

        # Concatenate side outputs along channels: (B,16*4,s,s,s)
        cat_side = torch.cat((s1, s2, s3, s4), dim=1)
                
        # added attention layer
        cat_side = self.to_patch_embedding(cat_side)
        instance_embedding = self.instance_embedding(torch.arange(self.config.num_pairs).to(x.device))
        instance_embedding = rearrange(instance_embedding, "p d -> 1 p d")
        cat_side = self.transformer(cat_side + instance_embedding)
        cat_side = self.to_conv_embedding(cat_side)
        
        # l5 aggregator => (B,100,s,s,s)
        out = self.model_surfacenet.l5(cat_side)

        # final 1-channel output
        out = self.model_surfacenet.out_conv(out)
        
        Y = rearrange(out, "(B P) 1 X Y Z -> B P 1 X Y Z", B=B, P=P)

        result = {
            "X": rearrange(cat_side, "(B P) C X Y Z -> B P C X Y Z", B=B, P=P),
            "Y": Y,
            "Y_surface": Y,
        }

        return result
