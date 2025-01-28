#! This is loosely based on https://github.com/lucidrains/vit-pytorch/blob/e7cba9ba6db1ad20e8d978b5f2c410d1257d2143/vit_pytorch/simple_vit_3d.py

from typing import List
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
from datasets import transforms
from datasets.transforms.point_transform import PointBasedTransformConfig
from networks.u_net import BasicConv3D, UNet3D, UNet3DConfig, deactivate_norm
from training.common import create_datamodule
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
    in_channels: int = 3
    dim_head: int = 64
    no_attn_feedthrough: bool = False
    disable_norm: bool = False

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

    def forward(self, x, attn_mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            
        else:
            raise NotImplementedError("Flash Attention is not available in this version of PyTorch")
        
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
    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, attn_mask) if self.no_attn_feedthrough else attn(x, attn_mask) + x
            x = ff(x) + x
        return self.norm(x)
    
class CostumEmbedding(nn.Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, dim)
    def forward(self, x):
        return self.embedding(x)


class PointTransformer(nn.Module):
    def __init__(self, config: PointTransformerConfig):
        super().__init__()
        self.mast3r_grid_size = torch.Tensor(config.grid_size)*config.grid_resolution / config.mast3r_grid_resolution
        self.config = config

        self.to_embedding = nn.Sequential(
            nn.LayerNorm(config.in_channels),
            nn.Linear(config.in_channels, config.dim),
            nn.LayerNorm(config.dim),
        )  
        
        # token used as voxel center -> later used for occupancy predication
        self.special_token = CostumEmbedding(1, config.dim)
        self.num_pts_embedding = CostumEmbedding((config.max_points_in_voxel + 1) - self.config.min_points_in_voxel, config.dim)
        
        if config.pair_matching == "first_centered":
            self.image_embedding = CostumEmbedding(config.seq_len, config.dim)
        else:
            raise NotImplementedError(f"Pair matching {config.pair_matching} not implemented")
        
        
        #self.transformer = Transformer(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, no_attn_feedthrough=config.no_attn_feedthrough)
        
        self.transformer = Transformer(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, no_attn_feedthrough=config.no_attn_feedthrough)
        
        dim_mlp = config.dim*config.max_points_in_voxel
        self.transformer = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_mlp, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_mlp),
        ) 
        
        self.to_occ = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim_mlp, 1),
        ) 
        
        if config.disable_norm:
            self.apply(deactivate_norm)
            
    def voxel_id_to_grid_id(self, voxel_ids):
        return voxel_ids // (self.mast3r_grid_size[2]*self.mast3r_grid_size[1]), (voxel_ids % (self.mast3r_grid_size[2]*self.mast3r_grid_size[1])) // self.mast3r_grid_size[2], (voxel_ids % (self.mast3r_grid_size[2]*self.mast3r_grid_size[1])) % self.mast3r_grid_size[2]

    def forward(self, x):
        """
        x: (batch_size, channels, depth, height, width)
        """
        feature_grid, image_id_grid, attn_mask, empty_grids, voxel_ids, voxel_counts, missing_pts = [x[key] for key in ["feature_grid", "image_id_grid", "attn_mask", "empty_grids", "voxel_ids", "voxel_counts", "missing_pts"]]

        # input dim is (num_occ_voxels, seq_len, channels)
        embeddings = self.to_embedding(feature_grid)
        
        embeddings = embeddings + self.image_embedding(image_id_grid)
        #embeddings = torch.cat([embeddings, self.special_token(torch.Tensor([0]).to(embeddings).int()).expand(num_occ_voxels, 1, -1)], dim=1)
        
        embeddings = embeddings + self.num_pts_embedding(missing_pts).unsqueeze(1)
        #attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1), dtype=bool, device=attn_mask.device)], dim=-1)
        
        if not self.config.visualize:
            embeddings = rearrange(embeddings, 'b n d -> b (n d)')
            embeddings = self.transformer(embeddings)#, attn_mask=attn_mask)
            special_token_embedding = embeddings#embeddings[:, 0, :]
            occ_pred = self.to_occ(special_token_embedding)
        else:
            occ_pred = attn_mask[:, :, :, 0].any(dim=-1)
        
        x, y, z = self.voxel_id_to_grid_id(voxel_ids)
        
        grid_index = torch.arange(
            voxel_counts.shape[0],
            dtype=torch.long,
            device=voxel_counts.device
        ).repeat_interleave(voxel_counts)
    
        occ_grids = torch.zeros((empty_grids.shape[0], *self.config.grid_size), dtype=torch.bfloat16, device=occ_pred.device)
        occ_grids[grid_index, x.long(), y.long(), z.long()] = occ_pred.squeeze(-1).to(occ_grids)
        loss_mask = torch.zeros((empty_grids.shape[0], *self.config.grid_size), dtype=bool, device=occ_pred.device)
        loss_mask[grid_index, x.long(), y.long(), z.long()] = True
        return occ_grids, loss_mask
    
if __name__ == "__main__":
    from training.mast3r.train_point_transformer import Config, DataConfig
    import pyvista as pv
    from visualization import occ

    data_config = DataConfig.load_from_files([
        "./config/data/base.yaml",
    ])
    
    data_config.batch_size = 8

    config = Config.load_from_files([
        "./config/trainer/base.yaml",
        "./config/module/base.yaml",
        "./config/network/point_transformer.yaml",
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels(),
    })
    
    config.visualize = True
    config.num_workers = 0
    config.val_num_workers = 0
    
    datamodule = create_datamodule(config, splits=["train", "val"], transform=transforms.PointBasedTransform, collate_fn=transforms.point_transform_collate_fn)
    datamodule.prepare_data()
    
    model = PointTransformer(config)
    
    colors = ["black", "red", "green", "blue"]
    
    for i, batch in enumerate(datamodule.train_dataloader()):
        feature_grid, image_id_grid, attn_mask, empty_grids, voxel_ids, voxel_counts, missing_pts = [batch["X"][key] for key in ["feature_grid", "image_id_grid", "attn_mask", "empty_grids", "voxel_ids", "voxel_counts", "missing_pts"]]
        points_of_grid = []
               
        occ_grids, loss_mask = model(batch["X"])
        
        for i in range(0, empty_grids.shape[0]):
            
            plotter = pv.Plotter(notebook=True)
            visualizer = occ.Visualizer(occ.Config(log_dir=".visualization", **config.model_dump()))
            visualizer.plotter = plotter
            
            voxel_grid_start_indices = voxel_counts.cumsum(0) - voxel_counts
            start_idx = voxel_grid_start_indices[i] 
            end_idx = voxel_grid_start_indices[i+1] if i+1 < voxel_grid_start_indices.shape[0] else voxel_counts.cumsum(0)
            #validity_mask = attn_mask[0]
            # get coordinates of the points using voxel ids
            coord = feature_grid[start_idx:end_idx, :, :3].reshape(-1, 3)
            image_id_grid_color = image_id_grid[start_idx:end_idx, :].reshape(-1)
            for j in range(image_id_grid_color.max()):
                points_in_voxel_grid= pv.PolyData(coord[image_id_grid_color == j].detach().cpu().numpy())
                plotter.add_points(points_in_voxel_grid, point_size=8, color=colors[j])

            #visualizer.add_from_occupancy_dict(batch["data"][i], opacity=0.5, to_world=False)
            
            #batch["data"][i]["occupancy_grid"] = loss_mask[i].unsqueeze(0)
            #visualizer.add_from_occupancy_dict(batch["data"][i], opacity=0.7, to_world=False)
            
            visualizer.export_html(f'../.visualization/point_cloud_visualization', timestamp=True)

