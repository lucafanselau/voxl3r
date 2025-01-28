import copy
from typing import List, Optional
from einops import rearrange
from loguru import logger
import torch
from torch import nn
from utils.config import BaseConfig
from jaxtyping import Float

import torch.nn.functional as F


class Simple3DUNetConfig(BaseConfig):
    in_channels: int
    base_channels: int

class UNet3DConfig(Simple3DUNetConfig):
    num_layers: int
    num_refinement_blocks: int
    refinement_bottleneck: int = 2
    skip_connections: bool
    disable_norm: bool = False
    with_downsampling: bool = True
    with_learned_pooling: bool = False
    keep_dim_during_up_conv: bool = False
    refinement_blocks: str = "simple"
    use_initial_batch_norm: bool = False
    # Only applies to skip connections
    # 1 means full dropout (eg. no skip connections), 0 means no dropout (full skip connections)
    skip_dropout_p: Optional[float] = None
    num_pairs: Optional[int] = None
    
def deactivate_norm(module):
    if isinstance(module, nn.BatchNorm3d):
        module.reset_parameters()
        module.eval()
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.zero_()
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
        module.eval()
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.zero_()
            
class BasicConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.gelu(x)
    
class InceptionBlockB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        
        self.branch = InceptionBlockA(in_channels, out_channels)
        
        self.with_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        if self.with_residual:
            return F.gelu(x + self.branch(x))
        else:
            return F.gelu(self.branch(x))
            
class InceptionBlockA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        
        out_1x1 = out_channels // 4
        out_3x3_reduced = out_channels // 4
        out_3x3 = out_channels // 2
        outinception_5x5_reduced = out_channels // 16
        out_5x5 = out_channels // 8
        out_pool = out_channels // 8


        self.branch1 = BasicConv3D(
            in_channels, out_1x1, kernel_size=1, stride=1
        )

        self.branch2 = nn.Sequential(
            BasicConv3D(in_channels, out_3x3_reduced, kernel_size=1),
            BasicConv3D(out_3x3_reduced, out_3x3, kernel_size=3, padding=1),
        )

        # Is in the original googLeNet paper 5x5 conv but in Inception_v2 it has shown to be
        # more efficient if you instead do two 3x3 convs which is what I am doing here!
        self.branch3 = nn.Sequential(
            BasicConv3D(in_channels, outinception_5x5_reduced, kernel_size=1),
            BasicConv3D(outinception_5x5_reduced, out_5x5, kernel_size=3, padding=1),
            BasicConv3D(out_5x5, out_5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BasicConv3D(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)

        return torch.cat([y1, y2, y3, y4], 1)

class Block1x1_3x3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int
    ) -> None:
        super().__init__()
    
        conv_3d = BasicConv3D
        latent_space = in_channels if in_channels > out_channels else out_channels
        self.branch1x1 = conv_3d(in_channels, latent_space, kernel_size=1)   
        self.branch3x3 = conv_3d(latent_space, out_channels, kernel_size=3, padding=1)

    def _forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        branch1x1_out = self.branch1x1(x)

        branch3x3_out = self.branch3x3(branch1x1_out)
        return branch3x3_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)
    
class Block3x3_1x1(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int
    ) -> None:
        super().__init__()
    
        conv_3d = BasicConv3D
        latent_space = in_channels if in_channels > out_channels else out_channels
        self.branch3x3 = conv_3d(in_channels, latent_space, kernel_size=3, padding=1)
        self.branch1x1 = conv_3d(latent_space, out_channels, kernel_size=1)    

    def _forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        branch3x3_out = self.branch3x3(x)
        
        branch1x1_out = self.branch1x1(branch3x3_out)
        return branch1x1_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)
    
class RefinementBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, padding=1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        return x
    
class RefinementBlockWithSkip(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, padding=1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.gelu = nn.GELU()
        
        self.with_skip = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (self.conv1(x) + x) if self.with_skip else self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, 
                 channels_in, 
                 num_downsampling_steps, 
                 num_refinement_layers,
                 refinement_block=RefinementBlock, 
                 channel_upscaling=RefinementBlock,
                 learned_downscaling=False, 
                 ):
        super().__init__()
        self.num_downsampling_steps = num_downsampling_steps
        self.num_refinement_layers = num_refinement_layers
        self.learned_downscaling = learned_downscaling
        
        self.layer_dim = []
        
        for i in range(0, self.num_downsampling_steps):
            self.layer_dim.append(channels_in * 2**i)
        
        logger.info(f"Building Encoder with layer dimensions: {self.layer_dim}")
        
        enc_layers = []
        
        for i in range(0, self.num_downsampling_steps):
            previous_layer_feature_dim = self.layer_dim[i-1] if i > 0 else self.layer_dim[i]
            layer_feature_dim = self.layer_dim[i]
            
            if i > 0:
                enc_layers.append([channel_upscaling(previous_layer_feature_dim, layer_feature_dim)])
            else:
                enc_layers.append([nn.Identity()])
                
            for _ in range(num_refinement_layers):
                enc_layers[i].extend([
                    refinement_block(layer_feature_dim, layer_feature_dim),
                    ])
        
        self.enc_conv_layers = nn.ModuleList([nn.Sequential(*layer) for layer in enc_layers])
        
        if self.learned_downscaling:
            enc_downsampling_layers = []
            for i in range(0, self.num_downsampling_steps):
                enc_downsampling_layers.append([
                        nn.Conv3d(self.layer_dim[i], self.layer_dim[i], 2, stride=2),
                        nn.BatchNorm3d(self.layer_dim[i]),
                        nn.GELU()
                        ])
            self.enc_downsampling = nn.ModuleList(enc_downsampling_layers)
        else:
            self.enc_downsampling = nn.MaxPool3d(2, stride=2)

        
    def forward(self, x):
        
        enc_layer_feature_map = []
        for i in range(0, self.num_downsampling_steps):
            x = self.enc_conv_layers[i](x)
            enc_layer_feature_map.append(x)
                
            if self.learned_downscaling:
                x = self.enc_downsampling[i](x)
            else:
                x = self.enc_downsampling(x)
        
        return x, enc_layer_feature_map
        
class DecoderBlock(nn.Module):
    def __init__(self, 
                 channels_in, 
                 num_upscaling_steps, 
                 num_refinement_blocks,
                 refinement_block=RefinementBlock, 
                 use_skip_connections=False,
                 keep_dim_after_skip=False,
                 skip_dropout=None,
                 ):
        super().__init__()
        self.use_skip_connections = use_skip_connections
        self.num_upsampling_steps = num_upscaling_steps
        self.num_refinement_blocks = num_refinement_blocks
        self.skip_dropout = skip_dropout
        
        if keep_dim_after_skip and not use_skip_connections:
            raise ValueError("keep_dim_after_skip can only be set to True if use_skip_connections is True")
        
        self.layer_dim = [channels_in]
        
        for i in range(0, self.num_upsampling_steps):
            self.layer_dim.append(self.layer_dim[-1]//2)
        
        logger.info(f"Building Encoder with layer dimensions: {self.layer_dim}")
        
        dec_refinement_layers = []
        dec_layers_upscaling = []
        
        for i in range(0, self.num_upsampling_steps):
            if keep_dim_after_skip and i > 0:
                i -= 1
            previous_layer_feature_dim = self.layer_dim[i]
            layer_feature_dim = self.layer_dim[i + 1]
            
            dec_layers_upscaling.append([
                    nn.ConvTranspose3d(previous_layer_feature_dim, layer_feature_dim, 2, stride=2),
                    nn.BatchNorm3d(layer_feature_dim),
                    nn.GELU(),
                    ])
            
            refinement_dim = 2*layer_feature_dim if keep_dim_after_skip else layer_feature_dim
                
            for j in range(num_refinement_blocks):
                if j == 0:
                    dec_refinement_layers.append([
                        refinement_block(2*layer_feature_dim if use_skip_connections else layer_feature_dim, refinement_dim),
                        ])
                else:
                    dec_refinement_layers[i].extend([
                        refinement_block(refinement_dim, refinement_dim),
                        ])
        
        self.dec_refinement_layers = nn.ModuleList([nn.Sequential(*layer) for layer in dec_refinement_layers])
        self.dec_layers_upscaling = nn.ModuleList([nn.Sequential(*layer) for layer in dec_layers_upscaling])

        
    def forward(self, x, enc_layer_feature_map = []):
        dec_layer_feature_map = []
        
        for i in range(0, self.num_upsampling_steps):
            x = self.dec_layers_upscaling[i](x)
            
            if self.use_skip_connections:
                x = torch.cat([x, self.skip_dropout(enc_layer_feature_map[-1-i]) if self.skip_dropout is not None else enc_layer_feature_map[-1-i]], dim=1)
            
            if self.num_refinement_blocks:
                x = self.dec_refinement_layers[i](x)
            
            dec_layer_feature_map.append(x)
        
        return x
    
class OccPreditor(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        if len(in_channels) > 1:
            self.conv = nn.ModuleList([nn.Conv3d(in_channels[i], 1, kernel_size=1) for i in range(len(in_channels))])
        else:
            self.conv = nn.Conv3d(in_channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            return [conv(x[i]) for i, conv in enumerate(self.conv)]
        else:
            return self.conv(x)
        
class_mapping = {
    "block1x1_3x3": Block1x1_3x3,
    "block3x3_1x1": Block3x3_1x1,
    "simple": RefinementBlock,
    "simpleWithSkip": RefinementBlockWithSkip,
    "inceptionBlockA": InceptionBlockA,
    "inceptionBlockB": InceptionBlockB
}    
        

class UNet3D(nn.Module):
    def __init__(self, config: UNet3DConfig, with_bottleneck: bool = True):
        super().__init__()
        self.config = config
        
        if config.use_initial_batch_norm:
            self.initial_batch_norm = nn.BatchNorm3d(config.in_channels)
        
        if config.refinement_blocks in class_mapping:
            encoder_refinement_block=class_mapping[config.refinement_blocks]
            decoder_refinement_block=class_mapping[config.refinement_blocks]
        else:
            raise ValueError(f"Refinement block {config.refinement_blocks} not found")
        
        dropout = None
        if config.skip_connections:
            assert config.skip_dropout_p is not None, "Skip dropout probability must be set if skip connections are used"
            dropout = nn.Dropout3d(config.skip_dropout_p)
            
        if config.skip_dropout_p == 1:
            config.skip_connections = False
            print("Setting skip connections to False because skip dropout probability is 1")
            
        if not config.with_downsampling and  config.in_channels != config.base_channels:
            raise ValueError("If no downsampling is used the number of layers must be equal to the base channels")
        
        if config.with_downsampling:
            self.downscaling_enc1 = nn.Sequential(
                BasicConv3D(config.in_channels, config.base_channels, kernel_size=1),
            )

        self.encoder = EncoderBlock(config.base_channels, config.num_layers, config.num_refinement_blocks, encoder_refinement_block, encoder_refinement_block)
        
        layer_dim_enc = self.encoder.layer_dim[-1]
        dec_in_dim = 2*layer_dim_enc

        self.with_bottleneck = with_bottleneck
        if with_bottleneck:
            if config.refinement_blocks in ["simple", "simpleWithSkip"]:
                self.bottleneck_layer_list = [encoder_refinement_block(layer_dim_enc, dec_in_dim, kernel_size=1, padding=0)]
                for _ in range(self.config.refinement_bottleneck):
                        self.bottleneck_layer_list.append(encoder_refinement_block(dec_in_dim, dec_in_dim, kernel_size=1, padding=0))

            else:
                self.bottleneck_layer_list = [BasicConv3D(layer_dim_enc, dec_in_dim, kernel_size=1)]
                for _ in range(self.config.refinement_bottleneck):
                        self.bottleneck_layer_list.append(BasicConv3D(dec_in_dim, dec_in_dim, kernel_size=1))

            self.bottleneck = nn.Sequential(*self.bottleneck_layer_list)
        

        self.decoder = DecoderBlock(config.num_pairs*dec_in_dim if config.num_pairs is not None else dec_in_dim, config.num_layers, config.num_refinement_blocks, decoder_refinement_block, config.skip_connections, config.keep_dim_during_up_conv, dropout)

        self.occ_layer_predictors_dim = []

        if self.config.keep_dim_during_up_conv:
            self.occ_layer_predictors_dim.append(self.decoder.layer_dim[-2])
        else:   
            self.occ_layer_predictors_dim.append(self.decoder.layer_dim[-1])
        
        self.occ_predictor = OccPreditor(self.occ_layer_predictors_dim)
        
        if config.disable_norm:
            self.apply(deactivate_norm)
                
    
    def bottleneck_forward(self, in_bottleneck: Float[torch.Tensor, "batch channels depth height width"], B, P):   
        in_dec = self.bottleneck_layer(in_bottleneck)
        return in_dec
    
    def forward(
        self, x: Float[torch.Tensor, "batch channels depth height width"]
    ) -> Float[torch.Tensor, "batch 1 depth height width"]:
        B, P, I, C, X, Y, Z = x.shape
        
        x = rearrange(x, "B P I C X Y Z -> (B P) (I C) X Y Z")
        
        if self.config.with_downsampling:
            x = self.downscaling_enc1(x)
        
        x, enc_feature_map = self.encoder(x)
        
        if self.with_bottleneck:
            x = self.bottleneck(x)
    
        x = self.decoder(x, enc_feature_map)
        occ = self.occ_predictor(x)
        return rearrange(occ, "(B P) 1 X Y Z -> B P 1 X Y Z", B=B, P=P)
   
   
def main():
    
    from training.mast3r import train
    
    data_config = train.DataConfig.load_from_files([
    "./config/data/base.yaml",
    "./config/data/undistorted_scenes.yaml"
    ])
    config = train.Config.load_from_files([
        "./config/trainer/tune.yaml",
        "./config/network/base_unet.yaml",
        "./config/network/unet3D.yaml",
        "./config/module/base.yaml"
    ], {
        **data_config.model_dump(),
        "in_channels": data_config.get_feature_channels()
    })

    config.skip_prepare = True
    config.num_refinement_blocks = 3
    config.refinement_blocks = "simple"
    config.num_layers = 3
    
    
    model = UNet3D(config) 
    
    model(torch.Tensor(2, 2, 48, 32, 32, 32))
    
if __name__ == "__main__":
    main()


