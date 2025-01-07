from typing import List, Optional
from einops import rearrange
import torch
from torch import nn
from utils.config import BaseConfig
from jaxtyping import Float

import torch.nn.functional as F


class Simple3DUNetConfig(BaseConfig):
    in_channels: int
    base_channels: int


class Simple3DUNet(nn.Module):
    def __init__(self, config: Simple3DUNetConfig):
        super().__init__()

        # Store config
        self.config = config

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(config.in_channels, config.base_channels, 1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(config.base_channels, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(config.base_channels * 2, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(config.base_channels * 2, config.base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 4),
            nn.ReLU(),
            nn.Conv3d(config.base_channels * 4, config.base_channels * 4, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 4),
            nn.ReLU(),
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(
                config.base_channels * 4, config.base_channels * 2, 2, stride=2
            ),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv3d(config.base_channels * 4, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
            nn.Conv3d(config.base_channels * 2, config.base_channels * 2, 3, padding=1),
            nn.BatchNorm3d(config.base_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose3d(
                config.base_channels * 2, config.base_channels, 2, stride=2
            ),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(config.base_channels * 2, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, config.base_channels, 3, padding=1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
            nn.Conv3d(config.base_channels, 1, 1),
        )

    def forward(
        self, x: Float[torch.Tensor, "batch channels depth height width"]
    ) -> Float[torch.Tensor, "batch 1 depth height width"]:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        # Decoder with skip connections
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        return dec1

class UNet3DConfig(Simple3DUNetConfig):
    num_layers: int
    refinement_layers: int
    refinement_bottleneck: int
    skip_connections: bool
    disable_norm: bool
    with_downsampling: bool
    with_learned_pooling: bool
    keep_dim_during_up_conv: bool
    refinement_blocks: str
    use_initial_batch_norm: bool = False
    # Only applies to skip connections
    # 1 means full dropout (eg. no skip connections), 0 means no dropout (full skip connections)
    skip_dropout_p: Optional[float] = None
    loss_layer_weights: list[float] = []
    num_pairs: int = None
    
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
        return F.relu(x, inplace=True)
    
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
            return F.relu(x + self.branch(x), inplace=True)
        else:
            return F.relu(self.branch(x), inplace=True)
            
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
        self, in_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, config: UNet3DConfig, with_bottleneck: bool = True):
        super().__init__()
        
        if config.use_initial_batch_norm:
            self.initial_batch_norm = nn.BatchNorm3d(config.in_channels)
        
        
        
        if config.refinement_blocks == "block1x1_3x3":
            refinement_block = Block1x1_3x3 
        elif config.refinement_blocks == "block3x3_1x1":
            refinement_block = Block3x3_1x1
        elif config.refinement_blocks == "simple":
            refinement_block = RefinementBlock
        elif config.refinement_blocks == "inceptionBlockA":
            refinement_block = InceptionBlockA
        elif config.refinement_blocks == "inceptionBlockB":
            refinement_block = InceptionBlockB
        else:
            raise ValueError("Unknown refinement block type")
        
        if config.skip_connections:
            assert config.skip_dropout_p is not None, "Skip dropout probability must be set if skip connections are used"
            self.dropout = nn.Dropout3d(config.skip_dropout_p)
            
        if config.skip_dropout_p == 1:
            config.skip_connections = False
            print("Setting skip connections to False because skip dropout probability is 1")
            
        if config.loss_layer_weights != [] and len(config.loss_layer_weights) != config.num_layers:
            raise ValueError("Loss layer weights must be empty or have the same length as the number of layers")

        self.config = config
        
        layer_dim = []
        
        if config.with_downsampling:
            self.downscaling_enc1 = nn.Sequential(
                BasicConv3D(config.in_channels, config.base_channels, kernel_size=1),
            )
        
            for i in range(0, self.config.num_layers + 1):
                layer_dim.append(config.base_channels * 2**i)
        
        else:
            
            for i in range(0, self.config.num_layers + 1):
                layer_dim.append(config.base_channels * 2**i)
            
            layer_dim[0] = config.in_channels
          
        enc_conv_layers = []
        
        for i in range(0, self.config.num_layers):
            previous_layer_feature_dim = layer_dim[i-1] if i > 0 else layer_dim[i]
            layer_feature_dim = layer_dim[i]
            if self.config.refinement_layers > 0:
                enc_conv_layers.append([
                    refinement_block(previous_layer_feature_dim, layer_feature_dim),
                ]) 
            for j in range(self.config.refinement_layers):
                enc_conv_layers[i].extend([
                    refinement_block(layer_feature_dim, layer_feature_dim),
                    ])
            
        self.enc_conv_layers = nn.ModuleList([nn.Sequential(*layer) for layer in enc_conv_layers])
        
        if config.num_pairs is not None:
            for i in range(len(layer_dim)):
                layer_dim[i] = layer_dim[i] * config.num_pairs
            
        
        if self.config.with_learned_pooling:
            enc_downsampling_layers = []
            for i in range(0, self.config.num_layers):
                if self.config.refinement_layers > 0:
                    enc_downsampling_layers.append([
                        nn.Conv3d(layer_dim[i], layer_dim[i], 2, stride=2),
                        nn.BatchNorm3d(layer_dim[i]),
                        nn.ReLU()
                        ])
                else:
                    enc_downsampling_layers.append([
                        nn.Conv3d(layer_dim[i], layer_dim[i+1], 2, stride=2),
                        nn.BatchNorm3d(layer_dim[i+1]),
                        nn.ReLU()
                        ])
                    
                
            self.enc_downsampling = nn.ModuleList([nn.Sequential(*layer) for layer in enc_downsampling_layers])
        else:
            self.enc_downsampling = nn.MaxPool3d(2, stride=2)
            
        if with_bottleneck:
            self.bottleneck_layer_list = [BasicConv3D(layer_dim[-2], layer_dim[-1], kernel_size=1)]
            
            if config.refinement_blocks == "simple":
                for _ in range(self.config.refinement_bottleneck):
                    self.bottleneck_layer_list.append(BasicConv3D(layer_dim[-1], layer_dim[-1], kernel_size=1))
            else:
                for _ in range(self.config.refinement_bottleneck):
                    self.bottleneck_layer_list.append(refinement_block(layer_dim[-1], layer_dim[-1]))
            
            self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
            
        dec_refinement_layers = []
        dec_up_convs = []
        
        for i in range(0, self.config.num_layers):
            previous_layer_feature_dim = layer_dim[-1-i] 
            layer_feature_dim = layer_dim[-2-i]
            if self.config.keep_dim_during_up_conv:
                dim_input = previous_layer_feature_dim if i == 0 else layer_dim[-i] 
                dec_up_convs.append([
                    nn.ConvTranspose3d(dim_input, dim_input, 2, stride=2),
                    nn.BatchNorm3d(dim_input),
                    nn.ReLU(),
                    ])
            else:
                dec_up_convs.append([
                    nn.ConvTranspose3d(previous_layer_feature_dim, layer_feature_dim, 2, stride=2),
                    nn.BatchNorm3d(layer_feature_dim),
                    nn.ReLU(),
                    ])
        
        for i in range(0, self.config.num_layers):
            if self.config.keep_dim_during_up_conv:
                layer_feature_dim = layer_dim[-1-i]
            else:
                layer_feature_dim = layer_dim[-i-2]
            
            if self.config.refinement_layers > 0:
                if self.config.skip_connections:
                    if self.config.keep_dim_during_up_conv:
                        dec_refinement_layers.append([
                            refinement_block((layer_dim[-1-i] if i == 0 else layer_dim[-i]) + layer_dim[- i - 2], layer_feature_dim),
                        ])
                    else:
                        dec_refinement_layers.append([
                            refinement_block(2*layer_feature_dim, layer_feature_dim),
                        ])
                else:
                    dec_refinement_layers.append([
                        refinement_block(layer_feature_dim, layer_feature_dim),
                    ])
                        
                for j in range(self.config.refinement_layers):
                    dec_refinement_layers[i].extend([
                        refinement_block(layer_feature_dim, layer_feature_dim),
                        ])
                
        self.dec_refinement_layers = nn.ModuleList([nn.Sequential(*layer) for layer in dec_refinement_layers])
        self.dec_up_convs = nn.ModuleList([nn.Sequential(*layer) for layer in dec_up_convs])
        
        if self.config.loss_layer_weights != []:
            self.occ_layer_predictors = []
            for i in range(self.config.num_layers):
                # we want to predict the occupancy for all layers including the bottleneck layer
                if i == 0:
                    self.occ_layer_predictors.append(nn.Conv3d(layer_dim[-i - 1], 1, 1))
                else:
                    if self.config.keep_dim_during_up_conv:
                        self.occ_layer_predictors.append(nn.Conv3d(layer_dim[-i], 1, 1))
                    else:
                        self.occ_layer_predictors.append(nn.Conv3d(layer_dim[-i-1], 1, 1))
                
            self.occ_layer_predictors = nn.ModuleList(self.occ_layer_predictors)

        if self.config.keep_dim_during_up_conv:
            self.occ_predictor = nn.Conv3d(layer_dim[1], 1, 1)
        else:   
            self.occ_predictor = nn.Conv3d(layer_dim[0], 1, 1)
        
        if config.disable_norm:
            self.apply(deactivate_norm)

        self.latent_dim = layer_dim[-1]
                

    def encoder_forward(self, in_enc: Float[torch.Tensor, "batch channels depth height width"]):
        
        if self.config.use_initial_batch_norm:
            in_enc = self.initial_batch_norm(in_enc)
            
        if self.config.with_downsampling:
            in_enc = self.downscaling_enc1(in_enc)

        enc_layer_out = []
        for i in range(0, self.config.num_layers):
            if self.config.refinement_layers > 0:
                in_enc = self.enc_conv_layers[i][0](in_enc)
                enc_layer_out.append(in_enc)
                
            if self.config.with_learned_pooling:
                in_enc = self.enc_downsampling[i](in_enc)
            else:
                in_enc = self.enc_downsampling(in_enc)
        
        return in_enc, enc_layer_out
    
    def bottleneck_forward(self, in_bottleneck: Float[torch.Tensor, "batch channels depth height width"], B, P):   
        in_dec = self.bottleneck_layer(in_bottleneck)
        
        occ_layer_out = []
        if self.config.loss_layer_weights != []:
            occ_layer_out.append(rearrange(self.occ_layer_predictors[0](in_dec), "(B P) 1 X Y Z -> B P X Y Z", B=B, P=P))
        
        return in_dec, occ_layer_out
    
    def decoder_forward(self, in_dec: Float[torch.Tensor, "batch channels depth height width"], occ_layer_out, enc_layer_out, B, P):

        for i in range(0, self.config.num_layers):
            in_dec = self.dec_up_convs[i](in_dec)
            
            if self.config.refinement_layers > 0:
                if self.config.skip_connections:
                    in_dec = self.dec_refinement_layers[i](torch.cat([in_dec, self.dropout(enc_layer_out[-1-i])], dim=1))
                else:
                    in_dec = self.dec_refinement_layers[i](in_dec)
                
                if self.config.loss_layer_weights != []:
                    if i < self.config.num_layers - 1:
                        occ_layer_out.append(rearrange(self.occ_layer_predictors[i+1](in_dec), "(B P) 1 X Y Z -> B P X Y Z", B=B, P=P))

        if self.config.num_pairs is not None:
            occ = self.occ_predictor(in_dec).squeeze(1)
        else:
            occ = rearrange(self.occ_predictor(in_dec), "(B P) 1 X Y Z -> B P X Y Z", B=B, P=P)
        
        return occ
    
    
    def forward(
        self, x: Float[torch.Tensor, "batch channels depth height width"]
    ) -> Float[torch.Tensor, "batch 1 depth height width"]:
        
        B, P, C, X, Y, Z = x.shape
        
        x = rearrange(x, "B P C X Y Z -> (B P) C X Y Z")
        
        x, enc_layer_out = self.encoder_forward(x)
        x, occ_layer_out = self.bottleneck_forward(x, B, P)
        occ = self.decoder_forward(x, occ_layer_out, enc_layer_out, B, P)
        
        return [occ, *occ_layer_out[::-1]]  if self.config.loss_layer_weights else occ
   
   
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

    config.skip_connections = False
    config.with_downsampling = False
    config.with_learned_pooling = True
    config.refinement_layers = 0
    
    
    model = UNet3D(config) 
    
    model(torch.Tensor(2, 2, 48, 32, 32, 32))
    
if __name__ == "__main__":
    main()


