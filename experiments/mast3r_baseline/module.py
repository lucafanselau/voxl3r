from dataclasses import dataclass

import pytorch_lightning as pl

from extern.mast3r.mast3r.model import AsymmetricMASt3R


@dataclass
class Mast3rBaselineConfig:
    model_name: str = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def load_model(model_name=None):
    weights_path = "naver/" + model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path)
    return model


class Mast3rBaselineLightningModule(pl.LightningModule):
    def __init__(self, config: Mast3rBaselineConfig):
        super().__init__()
        self.config = config

        self.model = load_model(self.config.model_name)
