from pathlib import Path
from typing import Literal, Optional
from beartype import beartype
from einops import repeat, reduce
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
import torch.nn
from tqdm import tqdm

from datasets import transforms_batched
from utils.config import BaseConfig


class BCEWeightedConfig(BaseConfig):
    loss_strategy: Literal["fixed", "exponential", "per_batch"]
    pos_weight: Optional[float] = None
    loss_exp_alpha: Optional[float] = 0.01


class BCEWeighted(torch.nn.Module):
    def __init__(self, config: BCEWeightedConfig):
        super().__init__()
        self.config = config

        C = 1
        pos_weight = torch.zeros((C, 1, 1, 1))
        self.register_buffer("pos_weight", pos_weight)
        self.pos_weight: Optional[Tensor]

        if config.loss_strategy == "fixed":
            self.pos_weight = torch.tensor(config.pos_weight).reshape(C, 1, 1, 1)
        if config.loss_strategy == "exponential":
            self.pos_weight = torch.Tensor([1.0]).reshape(C, 1, 1, 1)
            assert (
                config.loss_exp_alpha is not None
            ), "Loss weighting (alpha eg. 0.01) must be provided for exponential strategy"

    def get_batch_pos_weight(
        self, target: Bool[torch.Tensor, "B N 1 X Y Z"]
    ) -> Float[torch.Tensor, "1 1 1 1"]:
        W, H, D = target.shape[-3:]
        count_pos = target.sum(dim=(1, 2, 3, 4, 5)).float()
        VOLUME = W * H * D
        count_pos = count_pos.clamp(min=1)
        pos_weight = (VOLUME - count_pos) / (count_pos)
        return pos_weight[pos_weight.isfinite()].median().reshape(1, 1, 1, 1)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        target: Float[torch.Tensor, "B N 1 X Y Z"],
        output: Float[torch.Tensor, "B N 1 X Y Z"],
    ):

        if self.config.loss_strategy == "per_batch":
            self.pos_weight = self.get_batch_pos_weight(target)

        if self.config.loss_strategy == "exponential":
            alpha = self.config.loss_exp_alpha
            self.pos_weight = (
                1 - alpha
            ) * self.pos_weight + alpha * self.get_batch_pos_weight(target)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, target.to(output), pos_weight=self.pos_weight, reduction="none"
        )

        loss = reduce(loss, "B I ... -> B I", "mean")

        return loss
    
    
def main():
    
    from training.common import create_datamodule
    from training.mast3r.train_aggregator import DataConfig

    data_config = DataConfig.load_from_files(
    [
        "./config/data/base.yaml",
        "./config/data/mast3r.yaml",
        "./config/data/mast3r_transform.yaml",
        "./config/data/mast3r_transform_batched.yaml",
    ]
    )
    #data_config.scenes = ["39f36da05b", "5a269ba6fe", "dc263dfbf0"]
    
    data_config.scenes = [
        path.name
        for path in Path("/mnt/dorta/scannetpp/preprocessed").iterdir()
        if path.is_dir()
    ]

    data_config.enable_rotation = False
    data_config.num_workers = 11
    datamodule = create_datamodule(data_config, splits=["train"])
    datamodule.prepare_data()
    
    pos_weigth_accumulated = 0
    
    samplerConfig = data_config.model_copy()
    samplerConfig.split = None
    occGridSampler = transforms_batched.ComposeTransforms(samplerConfig)

    for batch in tqdm(iter(datamodule.train_dataloader())):
        occGridSampler(batch)
        target = batch["Y"] # shape B 1 W H D
        W, H, D = target.shape[-3:]
        count_pos = target.sum(dim=(1, 2, 3, 4)).float().mean()
        VOLUME = H*W*D
        pos_weight = (VOLUME - count_pos) / (count_pos)
        pos_weigth_accumulated += pos_weight
    # Mean pos_weight: 29.78364372253418             
    print(f"Mean pos_weight: {pos_weigth_accumulated/len(datamodule.train_dataloader())}")


if __name__ == "__main__":
    main()
