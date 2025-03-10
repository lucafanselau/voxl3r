from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import BaseConfig


class BaseSurfaceNetConfig(BaseConfig):
    """Base configuration for SurfaceNet architecture"""

    # Input configuration
    in_channels: int  # local features


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
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class SurfaceNet(nn.Module):
    """
    A PyTorch module implementing the core 3D SurfaceNet architecture
    based on Table 1 and Figure 3 of the paper.
    """

    def __init__(self, config):
        """
        in_channels = 48: local features
        """
        super(SurfaceNet, self).__init__()
        in_channels = config.in_channels

        mapping_conv = make_conv_block(in_channels, config.l1_dim, kernel_size=1, padding=0)

        # --- l1 block: 3 conv layers, each (3×3×3)
        l1_1 = make_conv_block(config.l1_dim, config.l1_dim)
        l1_2 = make_conv_block(config.l1_dim, config.l1_dim)
        l1_3 = make_conv_block(config.l1_dim, config.l1_dim)

        self.l1 = nn.Sequential(*[mapping_conv, l1_1, l1_2, l1_3])

        # side layer s1
        s1_conv = nn.Conv3d(config.l1_dim, config.side_dim, kernel_size=1, bias=False)
        s1_bn = nn.BatchNorm3d(config.side_dim)

        self.s1 = nn.Sequential(*[s1_conv, s1_bn])

        # p1: 3D max pooling
        self.p1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # --- l2 block: 3 conv layers
        l2_1 = make_conv_block(config.l1_dim, config.l2_dim)
        l2_2 = make_conv_block(config.l2_dim, config.l2_dim)
        l2_3 = make_conv_block(config.l2_dim, config.l2_dim)

        self.l2 = nn.Sequential(*[l2_1, l2_2, l2_3])

        # side layer s2
        s2_conv = nn.ConvTranspose3d(
            config.l2_dim, config.side_dim, kernel_size=2, stride=2, bias=False
        )
        s2_bn = nn.BatchNorm3d(config.side_dim)

        self.s2 = nn.Sequential(*[s2_conv, s2_bn])

        # p2
        self.p2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # --- l3 block
        l3_1 = make_conv_block(config.l2_dim, config.l3_dim)
        l3_2 = make_conv_block(config.l3_dim, config.l3_dim)
        l3_3 = make_conv_block(config.l3_dim, config.l3_dim)

        self.l3 = nn.Sequential(*[l3_1, l3_2, l3_3])

        # side layer s3
        s3_conv = nn.ConvTranspose3d(
            config.l3_dim, config.side_dim, kernel_size=4, stride=4, bias=False
        )
        s3_bn = nn.BatchNorm3d(config.side_dim)

        self.s3 = nn.Sequential(*[s3_conv, s3_bn])

        # --- l4 block (dilated conv)
        l4_1 = make_conv_block(config.l3_dim, config.l4_dim, dilation=1)
        l4_2 = make_conv_block(config.l4_dim, config.l4_dim, dilation=1)
        l4_3 = make_conv_block(config.l4_dim, config.l4_dim, dilation=1)

        self.l4 = nn.Sequential(*[l4_1, l4_2, l4_3])

        # side layer s4
        s4_conv = nn.ConvTranspose3d(
            config.l4_dim, config.side_dim, kernel_size=4, stride=4, bias=False
        )
        s4_bn = nn.BatchNorm3d(config.side_dim)

        self.s4 = nn.Sequential(*[s4_conv, s4_bn])

        # --- final aggregator l5
        l5_1 = make_conv_block(config.side_dim * 4, config.l5_dim, kernel_size=3)
        l5_2 = make_conv_block(config.l5_dim, config.l5_dim, kernel_size=3)

        self.l5 = nn.Sequential(*[l5_1, l5_2])

        # output
        self.out_conv = nn.Conv3d(config.l5_dim, 1, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm3d(1)

    def forward(self, batch):
        """
        x is a 5D tensor [B, 6, s, s, s].
        """
        x = batch["X"]
        B, P, I, C, X, Y, Z = x.shape
        x = rearrange(x, "B P I C X Y Z -> (B P) (I C) X Y Z")

        # --- l1
        x1 = self.l1(x)

        # side 1
        s1 = self.s1(x1)
        s1 = torch.sigmoid(s1)  # side outputs use sigmoid

        # pooling
        p1 = self.p1(x1)

        # --- l2
        x2 = self.l2(p1)

        # side 2
        s2 = self.s2(x2)
        s2 = torch.sigmoid(s2)

        # pooling
        p2 = self.p2(x2)

        # --- l3
        x3 = self.l3(p2)

        # side 3
        s3 = self.s3(x3)
        s3 = torch.sigmoid(s3)

        # --- l4 (dilated)
        x4 = self.l4(x3)

        # side 4
        s4 = self.s4(x4)
        s4 = torch.sigmoid(s4)

        # Concatenate side outputs along channels: (B,16*4,s,s,s)
        cat_side = torch.cat((s1, s2, s3, s4), dim=1)

        # l5 aggregator => (B,100,s,s,s)
        out = self.l5(cat_side)

        # final 1-channel output
        out = self.out_conv(out)
        out = self.out_bn(out)

        result = {
            "X": rearrange(cat_side, "(B P) C X Y Z -> B P C X Y Z", B=B, P=P),
            "Y": rearrange(out, "(B P) 1 X Y Z -> B P 1 X Y Z", B=B, P=P),
        }

        return result
