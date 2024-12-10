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

