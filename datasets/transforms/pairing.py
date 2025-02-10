


from pathlib import Path

import torch
from .images import StackTransformations
from utils.config import BaseConfig
from datasets.chunk.pair_matching import PairMatching
from torch import nn

class PairingTransformConfig(BaseConfig):
    pair_matching: str = "first_centered"

class PairingTransform(nn.Module):
    def __init__(self, config: PairingTransformConfig, *_args):
        super().__init__()
        self.config = config

        self.stack_transformations = StackTransformations()

        self.pair_matching = PairMatching[self.config.pair_matching]()
        
    def forward(self, data: dict) -> dict:
        seq_len = len(data["images_tensor"])
        pair_indices = self.pair_matching(seq_len)

        image_dict = {
                Path(key).name: value
                for key, value in zip(data["images"], data["cameras"])
        }
        new_shape = torch.tensor(data["images_tensor"].shape[-2:]).float()
        transformations, T_cw, _ = self.stack_transformations(image_dict, new_shape=new_shape)

        data["paired"] = {
            "images": data["images_tensor"][pair_indices],
            "transformations": transformations[pair_indices],
            "T_cw": T_cw[pair_indices],
            "indices": pair_indices,
        }

        data["X"] = data["images_tensor"][pair_indices]
        data["type"] = "images"

        return data

        

