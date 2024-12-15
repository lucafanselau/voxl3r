from einops import rearrange
import torch
from torch import nn
from utils.config import BaseConfig
from jaxtyping import Float


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
    skip_connections: bool
    
    
def deactivate_batchnorm(module):
    if isinstance(module, nn.BatchNorm2d):
        module.reset_parameters()
        module.eval()
        with torch.no_grad():
            module.weight.fill_(1.0)
            module.bias.zero_()

class UNet3D(nn.Module):
    def __init__(self, config: UNet3DConfig):
        super().__init__()

        self.config = config
        
        self.downscaling_enc1 = nn.Sequential(
            nn.Conv3d(config.in_channels, config.base_channels, 1),
            nn.BatchNorm3d(config.base_channels),
            nn.ReLU(),
        )
          
        enc_conv_layers = []
        
        layer_dim = []
        
        for i in range(0, self.config.num_layers + 1):
            layer_dim.append(config.base_channels * 2**i)
        
        for i in range(0, self.config.num_layers):
            previous_layer_feature_dim = layer_dim[i-1] if i > 0 else layer_dim[i]
            layer_feature_dim = layer_dim[i]
            enc_conv_layers.append([
                nn.Conv3d(previous_layer_feature_dim, layer_feature_dim, 3, padding=1),
                nn.BatchNorm3d(layer_feature_dim),
                nn.ReLU(),
            ]) 
            for j in range(self.config.refinement_layers):
                enc_conv_layers[i].extend([
                    nn.Conv3d(layer_feature_dim, layer_feature_dim, 3, padding=1),
                    nn.BatchNorm3d(layer_feature_dim),
                    nn.ReLU()
                    ])
            
        self.enc_conv_layers = nn.ModuleList([nn.Sequential(*layer) for layer in enc_conv_layers])
        self.enc_downsampling = nn.MaxPool3d(2, stride=2)
        
        self.bottleneck_layer = nn.Sequential(
            nn.Conv3d(layer_dim[-2], layer_dim[-1], 1),
            nn.BatchNorm3d(layer_dim[-1],),
            nn.ReLU(),
        )
        
        dec_refinement_layers = []
        dec_up_convs = []
        
        for i in range(0, self.config.num_layers):
            previous_layer_feature_dim = layer_dim[self.config.num_layers - i]
            layer_feature_dim = layer_dim[self.config.num_layers - i - 1]
            dec_up_convs.append([
                nn.ConvTranspose3d(previous_layer_feature_dim, layer_feature_dim, 2, stride=2),
                nn.BatchNorm3d(layer_feature_dim),
                nn.ReLU(),
                ])
        
        for i in range(0, self.config.num_layers):
            layer_feature_dim = layer_dim[self.config.num_layers - i - 1] if (self.config.num_layers - i) > 1 else layer_dim[self.config.num_layers - i - 1]
            
            if self.config.skip_connections:
                dec_refinement_layers.append([
                    nn.Conv3d(2*layer_feature_dim, layer_feature_dim, 3, padding=1),
                    nn.BatchNorm3d(layer_feature_dim),
                    nn.ReLU()
                ])
            else:
                dec_refinement_layers.append([
                    nn.Conv3d(layer_feature_dim, layer_feature_dim, 3, padding=1),
                    nn.BatchNorm3d(layer_feature_dim),
                    nn.ReLU()
                ])
                    
            for j in range(self.config.refinement_layers):
                dec_refinement_layers[i].extend([
                    nn.Conv3d(layer_feature_dim, layer_feature_dim, 3, padding=1),
                    nn.BatchNorm3d(layer_feature_dim),
                    nn.ReLU()
                    ])
                
        self.dec_refinement_layers = nn.ModuleList([nn.Sequential(*layer) for layer in dec_refinement_layers])
        self.dec_up_convs = nn.ModuleList([nn.Sequential(*layer) for layer in dec_up_convs])
        
        self.occ_predictor = nn.Conv3d(config.base_channels, 1, 1)

                

    def forward(
        self, x: Float[torch.Tensor, "batch channels depth height width"]
    ) -> Float[torch.Tensor, "batch 1 depth height width"]:
        
        B, P, C, X, Y, Z = x.shape
        in_enc = self.downscaling_enc1(rearrange(x, "B P C X Y Z -> (B P) C X Y Z"))
        enc_layer_out = []
        for i in range(0, self.config.num_layers):
            in_enc = self.enc_conv_layers[i](in_enc)
            enc_layer_out.append(in_enc)
            in_enc = self.enc_downsampling(in_enc)
        
        dec_in = self.bottleneck_layer(in_enc)
        
        dec_layer_out = []
        for i in range(0, self.config.num_layers):
            dec_in = self.dec_up_convs[i](dec_in)
            if self.config.skip_connections:
                dec_in = self.dec_refinement_layers[i](torch.cat([dec_in, enc_layer_out[self.config.num_layers-1-i]], dim=1))
            else:
                dec_in = self.dec_refinement_layers[i](dec_in)
            dec_layer_out.append(dec_in)

        return rearrange(self.occ_predictor(dec_in), "(B P) 1 X Y Z -> B P X Y Z", B=B, P=P)
   
   
def main():
    config = UNet3DConfig(in_channels=96, base_channels=32, num_layers=2, refinement_layers=2, skip_connections=False)
    model = UNet3D(config) 
    
    model(torch.Tensor(4, 96, 32, 32, 32))
    
if __name__ == "__main__":
    main()


