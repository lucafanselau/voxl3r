#! This is loosely based on https://github.com/lucidrains/vit-pytorch/blob/e7cba9ba6db1ad20e8d978b5f2c410d1257d2143/vit_pytorch/simple_vit_3d.py

from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from datasets.transforms.point_based import PointBasedTransformConfig
from networks.u_net import BasicConv3D, UNet3D, UNet3DConfig, deactivate_norm
from utils.config import BaseConfig

# helpers

def triplet(t):
    return t if isinstance(t, tuple) else (t, t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

# classes
class PointTransformerConfig(PointBasedTransformConfig):
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    channels: int = 3
    dim_head: int = 64
    num_pairs: int = 4
    no_attn_feedthrough: bool = True
    use_learned_pe: bool = False

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.attend = nn.Softmax(dim = -1)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
            
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
   

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, no_attn_feedthrough = True):
        super().__init__()
        self.no_attn_feedthrough = no_attn_feedthrough
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) if self.no_attn_feedthrough else attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class PointTransformer(nn.Module):
    def __init__(self, config: PointTransformerConfig):
        super().__init__()
        self.pts_grid_size = torch.Tensor(config.grid_size)*config.grid_resolution / config.mast3r_grid_resolution
        patch_dim = config.channels
        num_patches = self.pts_grid_size[0]*self.pts_grid_size[1]*self.pts_grid_size[2] 

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, config.dim),
        )  
        
        # token used as voxel center -> later used for occupancy predication
        self.special_token = torch.nn.Embedding(1, config.dim)
        
        self.transformer = Transformer(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, no_attn_feedthrough=config.no_attn_feedthrough)
        
        if config.disable_norm:
            self.apply(deactivate_norm)

    def forward(self, x):
        """
        x: (batch_size, num_pairs, channels, depth, height, width)
        """
        return x