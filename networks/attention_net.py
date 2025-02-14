from functools import partial
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from grpc import Channel
from loguru import logger
import torch
import torch.nn as nn

from networks.point_transformer import CostumEmbedding
from utils.config import BaseConfig

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
    
def check_flex_attention():
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        return True
    except ImportError:
        return False

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, seq_len=None, num_pairs=4):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        
        self.has_flex = check_flex_attention()
        if self.has_flex:
            from torch.nn.attention.flex_attention import and_masks, create_block_mask, flex_attention
            grid_edge = int(math.cbrt(seq_len // num_pairs))
            threshold = 5

            def distance_3d_mask(b, h, q_idx, kv_idx):
                # this is in distance in voxel units (3d space)
                # in this configuration b.shape == []
                # we want the distance between q_idx and kv_idx to be less than or equal to threshold

                # the distance calculation should be in 3d though
                # so we need to convert the 1d indices to 3d indices (X Y Z) -> X Y Z

                # remove the pairs counter for the indices
                q_idx = q_idx // num_pairs
                kv_idx = kv_idx // num_pairs

                q_idx_x, q_idx_y, q_idx_z = q_idx // (grid_edge * grid_edge), (q_idx // grid_edge) % grid_edge, q_idx % grid_edge
                kv_idx_x, kv_idx_y, kv_idx_z = kv_idx // (grid_edge * grid_edge), (kv_idx // grid_edge) % grid_edge, kv_idx % grid_edge

                # distance in 3d (eg. norm2)
                norm = (q_idx_x - kv_idx_x) ** 2 + (q_idx_y - kv_idx_y) ** 2 + (q_idx_z - kv_idx_z) ** 2
                distance = torch.sqrt(norm)
                return distance <= threshold

            # HERE WE CAN ALSO CONTROL BLOCK_SIZE eg. the size in which sparsity is applied
            logger.debug(f"Creating attention mask with grid_edge {grid_edge}")
            BLOCK_SIZE = 128
            self.attn_mask = create_block_mask(distance_3d_mask, B=None, H=self.heads, Q_LEN=seq_len, KV_LEN=seq_len, BLOCK_SIZE=BLOCK_SIZE, _compile=True)

            def relative_positional_3d(score, b, h, q_idx, kv_idx):

                # remove the pairs counter for the indices
                q_idx = q_idx // num_pairs
                kv_idx = kv_idx // num_pairs

                q_idx_x, q_idx_y, q_idx_z = q_idx // (grid_edge * grid_edge), (q_idx // grid_edge) % grid_edge, q_idx % grid_edge
                kv_idx_x, kv_idx_y, kv_idx_z = kv_idx // (grid_edge * grid_edge), (kv_idx // grid_edge) % grid_edge, kv_idx % grid_edge

                # distance in 3d (eg. norm2)
                norm = (q_idx_x - kv_idx_x) ** 2 + (q_idx_y - kv_idx_y) ** 2 + (q_idx_z - kv_idx_z) ** 2
                distance = torch.sqrt(norm)
                distance = torch.clamp(distance, min=0.5)

                return score + (threshold/distance)

            
            self.flex_attention = torch.compile(partial(flex_attention, block_mask=self.attn_mask, score_mod=relative_positional_3d))

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and not self.has_flex:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.attend = nn.Softmax(dim = -1)

    def forward(self, x, attn_mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.has_flex:
            # efficient attention using Flex Attention 
            out = self.flex_attention(q, k, v)
        elif self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            
        else:
            raise NotImplementedError("Flash Attention is not available in this version of PyTorch")
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
   

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, no_attn_feedthrough = True, seq_len=None, num_pairs=4):
        super().__init__()
        self.no_attn_feedthrough = no_attn_feedthrough
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, seq_len=seq_len, num_pairs=num_pairs),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, attn_mask) if self.no_attn_feedthrough else attn(x, attn_mask) + x
            x = ff(x) + x
        return self.norm(x)
    
class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, dim)
    def forward(self, x):
        return self.embedding(x)


class AttentionNetConfig(BaseConfig):


    per_voxel_channels: int = 64
    num_image_pairs: int = 4

    dim: int = 512
    depth: int = 6
    heads: int = 8
    mlp_dim: int = 512
    dim_head: int = 64

    grid_size_sample: list[int]

class AttentionNet(nn.Module):
    def __init__(self, config: AttentionNetConfig):
        super().__init__()
        self.config = config

        X, Y, Z = config.grid_size_sample
        px, py, pz = (2, 2, 2)

        num_patches = (X // px) * (Y // py) * (Z // pz) * config.num_image_pairs
        patch_dim = config.per_voxel_channels * px * py * pz
        p = config.num_image_pairs


        self.to_patch_embedding = nn.Sequential(
            Rearrange('b p c (X px) (Y py) (Z pz) -> b p (X Y Z) (px py pz c)', px = px, py = py, pz = pz),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, config.dim),
            nn.LayerNorm(config.dim),
        )

        self.to_out_embedding = nn.Sequential(
            nn.Linear(config.dim, patch_dim),
            Rearrange('b (X Y Z p) (px py pz c) -> b p c (X px) (Y py) (Z pz)', p=p, X=X//px, Y=Y//py, Z=Z//pz, px = px, py = py, pz = pz),
        )

        self.transformer = Transformer(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, seq_len=num_patches, no_attn_feedthrough=False, num_pairs=config.num_image_pairs)

        self.out_conv = nn.Conv3d(config.per_voxel_channels * config.num_image_pairs, 1, kernel_size=1, bias=False)

        # instance embedding for each pair
        self.instance_embedding = nn.Embedding(config.num_image_pairs, config.dim)

    def forward(self, previous):
        """
        x: (batch_size, channels, depth, height, width)
        """

        x = previous["X"]
        # x.shape = torch.Size([16, 4, 64, 32, 32, 32]) (batch, pairs, channels, X, Y, Z)

        # transform to (batch, (pairs, channels), X, Y, Z)
        

        # transform to patches (batch, (pairs * channels * (2**3)), X/2, Y/2, Z/2)

        x = self.to_patch_embedding(x)
        # x is now (batch, (pairs * channels * (2**3)), dim)

        # add instance embedding
        instance_embedding = self.instance_embedding(torch.arange(self.config.num_image_pairs).to(x.device))
        instance_embedding = rearrange(instance_embedding, "p d -> 1 p 1 d")

        x = x + instance_embedding

        x = rearrange(x, "b p XYZ d -> b (XYZ p) d")
        # apply transformer
        x = self.transformer(x)

        # now let's go back to the original shape
        x = self.to_out_embedding(x)
        x = rearrange(x, "b p c x y z -> b (p c) x y z")

        x = self.out_conv(x)

        return {
            "Y": rearrange(x, "B 1 X Y Z -> B 1 1 X Y Z") # to be inline with the output of the SurfaceNet
        }
        
    