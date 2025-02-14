from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv_block(in_ch, out_ch, kernel_size=3, dilation=1):
    """
    Creates a 3D convolution -> batchnorm -> ReLU block.
    """
    padding = dilation  # a typical choice so that spatial size stays the same
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )

class SurfaceNet(nn.Module):
    """
    A PyTorch module implementing the core 3D SurfaceNet architecture
    based on Table 1 and Figure 3 of the paper.
    """
    def __init__(self, config):
        """
        in_channels = 6: two RGB-CVC volumes stacked along channel dim => (3 + 3).
        """
        super(SurfaceNet, self).__init__()
        in_channels = config.in_channels
        # --- l1 block: 3 conv layers, each (3×3×3), output size (32, s, s, s)
        l1_1 = make_conv_block(in_channels, 32)
        l1_2 = make_conv_block(32, 32)
        l1_3 = make_conv_block(32, 32)
        
        self.l1 = nn.Sequential(
            *[l1_1, l1_2, l1_3]
        )
        
        # side layer s1 => (16, s, s, s) after upsample.  
        # Just use 1×1 conv + BN + sigmoid, but we do it in two steps:
        s1_conv = nn.Conv3d(32, 16, kernel_size=1, bias=False)
        s1_bn   = nn.BatchNorm3d(16)
        
        self.s1 = nn.Sequential(
            *[s1_conv, s1_bn]
        )
        
        # p1: 3D max pooling => (32, s/2, s/2, s/2)
        self.p1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # --- l2 block: 3 conv layers => output size (80, s/2, s/2, s/2)
        l2_1 = make_conv_block(32, 80)
        l2_2 = make_conv_block(80, 80)
        l2_3 = make_conv_block(80, 80)
        
        self.l2 = nn.Sequential(
            *[l2_1, l2_2, l2_3]
        )
        
        # side layer s2 => (16, s, s, s)
        s2_conv = nn.ConvTranspose3d(80, 16, kernel_size=2, stride=2, bias=False)
        s2_bn   = nn.BatchNorm3d(16)
        
        self.s2 = nn.Sequential(
            *[s2_conv, s2_bn]
        )
        
        # p2 => (80, s/4, s/4, s/4)
        self.p2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # --- l3 block => (160, s/4, s/4, s/4)
        l3_1 = make_conv_block(80, 160)
        l3_2 = make_conv_block(160, 160)
        l3_3 = make_conv_block(160, 160)
        
        self.l3 = nn.Sequential(
            *[l3_1, l3_2, l3_3]
        )
        
        # side layer s3 => (16, s, s, s)
        s3_conv = nn.ConvTranspose3d(160, 16, kernel_size=4, stride=4, bias=False)
        s3_bn   = nn.BatchNorm3d(16)
        
        self.s3 = nn.Sequential(
            *[s3_conv, s3_bn]
        )
        
        # --- l4 block (dilated conv): (300, s/4, s/4, s/4)
        # repeated 3 times with dilation=2
        l4_1 = make_conv_block(160, 300, dilation=1)
        l4_2 = make_conv_block(300, 300, dilation=1)
        l4_3 = make_conv_block(300, 300, dilation=1)
        
        self.l4 = nn.Sequential(
            *[l4_1, l4_2, l4_3]
        )
        
        # side layer s4 => (16, s, s, s)
        s4_conv = nn.ConvTranspose3d(300, 16, kernel_size=4, stride=4, bias=False)
        s4_bn   = nn.BatchNorm3d(16)
        
        self.s4 = nn.Sequential(
            *[s4_conv, s4_bn]
        )

        # --- final aggregator l5 => (100, s, s, s)
        l5_1 = make_conv_block(16*4, 100, kernel_size=3)
        l5_2 = make_conv_block(100, 100, kernel_size=3)
        
        self.l5 = nn.Sequential(
            *[l5_1, l5_2]
        )

        # output => (1, s, s, s)
        self.out_conv = nn.Conv3d(100, 1, kernel_size=1, bias=False)
        self.out_bn   = nn.BatchNorm3d(1)

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
            "Y": rearrange(out, "(B P) 1 X Y Z -> B P 1 X Y Z", B=B, P=P)
        }
        
        return result
