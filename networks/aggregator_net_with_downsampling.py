from einops import rearrange
import torch
import torch.nn as nn

def make_conv_block(in_ch, out_ch, kernel_size=3, dilation=1):
    """
    Creates a 3D convolution -> InstanceNorm3d -> ReLU block.
    Padding is set to keep the spatial dimensions unchanged.
    """
    padding = dilation  # ensures output spatial size remains the same
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.ReLU(inplace=True)
    )

class AggregatorNetWithDownsampling(nn.Module):
    """
    A simple aggregator network that processes concatenated side outputs.c
    Expected input:
        x: A 5D tensor of shape (B, 16*4, D, H, W), where 16*4 equals 64 channels.
    
    Output:
        A tensor of shape (B, 100, D, H, W) after applying two convolutional blocks.
    """
    def __init__(self, *_args, **_kwargs):
        super(AggregatorNetWithDownsampling, self).__init__()
        # self.l5 = nn.Sequential(
        #     # voxel pairs communicate with each other
        #     nn.Conv3d(4*(16*4), 512, 1, bias=True),
        #     nn.InstanceNorm3d(512, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(512, 1024, 1, bias=True),
        #     nn.InstanceNorm3d(1024, affine=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(1024, 256, 1, bias=True),
        #     nn.InstanceNorm3d(256, affine=True),
        #     nn.ReLU(inplace=True),
        #     make_conv_block(256, 256, kernel_size=3),
        #     make_conv_block(256, 256, kernel_size=3),
        #     make_conv_block(256, 256, kernel_size=3),
        # )
        
        self.l5 = nn.Sequential(
            # voxel pairs communicate with each other
            make_conv_block(5*(16*4), 256, kernel_size=3),
            
            nn.Conv3d(256, 1024, 1, bias=True),
            nn.InstanceNorm3d(1024, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(1024, 256, 1, bias=True),
            nn.InstanceNorm3d(256, affine=True),
            nn.ReLU(inplace=True),
            
            make_conv_block(256, 256, kernel_size=3),
        )
        
        self.out_conv = nn.Conv3d(256, 1, kernel_size=1, bias=False)
    
    def forward(self, input):
        # B, P, C, X, Y, Z = input["X"].shape
        rearrange_in = rearrange(input["X"], "B P C X Y Z -> B (P C) X Y Z")
        out = self.l5(rearrange_in)
        
        # final 1-channel output
        out = self.out_conv(out)
        
 
        input["Y"] = rearrange(out, "B 1 X Y Z -> B 1 1 X Y Z") # to be inline with the output of the SurfaceNet
        
        return input
        
