from einops import rearrange
import torch
import torch.nn as nn

def make_conv_block(in_ch, out_ch, kernel_size=3, dilation=1):
    """
    Creates a 3D convolution -> BatchNorm -> ReLU block.
    Padding is set to keep the spatial dimensions unchanged.
    """
    padding = dilation  # ensures output spatial size remains the same
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True)
    )

class AggregatorNet(nn.Module):
    """
    A simple aggregator network that processes concatenated side outputs.
    
    Expected input:
        x: A 5D tensor of shape (B, 16*4, D, H, W), where 16*4 equals 64 channels.
    
    Output:
        A tensor of shape (B, 100, D, H, W) after applying two convolutional blocks.
    """
    def __init__(self, *_args, **_kwargs):
        super(AggregatorNet, self).__init__()
        self.l5 = nn.Sequential(
            make_conv_block(4*16*4, 256, kernel_size=3),
            nn.Conv3d(256, 512, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            make_conv_block(512, 128, kernel_size=3),
        )
        
        self.out_conv = nn.Conv3d(128, 1, kernel_size=1, bias=False)
        # self.out_bn   = nn.BatchNorm3d(1)
    
    def forward(self, input):
        # B, P, C, X, Y, Z = input["X"].shape
        out = self.l5(rearrange(input["X"], "B P C X Y Z -> B (P C) X Y Z"))
        
        # final 1-channel output
        out = self.out_conv(out)
        # out = self.out_bn(out)
        
        result = {
            "Y": rearrange(out, "B 1 X Y Z -> B 1 1 X Y Z") # to be inline with the output of the SurfaceNet
        }
        
        return result
        
