#! This is loosely based on https://github.com/lucidrains/vit-pytorch/blob/e7cba9ba6db1ad20e8d978b5f2c410d1257d2143/vit_pytorch/simple_vit_3d.py

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
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
class VolumeTransformerConfig(BaseConfig):
    image_size: int
    image_patch_size: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    channels: int = 3
    dim_head: int = 64

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

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))


    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
   

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class VolumeTransformer(nn.Module):
    def __init__(self, config: VolumeTransformerConfig):
        super().__init__()
        image_height, image_width, image_depth = triplet(config.image_size)
        patch_height, patch_width, patch_depth = triplet(config.image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = config.channels * patch_height * patch_width * patch_depth

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d pd) (h p1) (w p2) -> b d h w (p1 p2 pd c)', p1 = patch_height, p2 = patch_width, pd = patch_depth),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )

        self.transformer = Transformer(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim)

    def forward(self, volume):
        """
        volume: (batch_size, channels, depth, height, width)
        """
        # *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(volume)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)

        return x