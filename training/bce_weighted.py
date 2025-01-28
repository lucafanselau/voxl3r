from typing import Literal, Optional
from beartype import beartype
from einops import repeat
from jaxtyping import jaxtyped, Float, Bool
from torch import Tensor
import torch.nn

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
            assert config.loss_exp_alpha is not None, "Loss weighting (alpha eg. 0.01) must be provided for exponential strategy"


    def get_batch_pos_weight(self, target: Bool[torch.Tensor, "B N 1 X Y Z"]) -> Float[torch.Tensor, "1 1 1 1"]:
        W, H, D = target.shape[-3:]
        count_pos = target.sum(dim=(1, 2, 3, 4, 5)).float()
        VOLUME = W * H * D
        pos_weight = (VOLUME - count_pos) / count_pos
        return pos_weight[pos_weight.isfinite()].mean().reshape(1, 1, 1, 1)
    
    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        target: Float[torch.Tensor, "B N 1 X Y Z"],
        output: Float[torch.Tensor, "B N 1 X Y Z"]
    ):
        
        if self.config.loss_strategy == "per_batch":
            self.pos_weight = self.get_batch_pos_weight(target)

        if self.config.loss_strategy == "exponential":
            alpha = self.config.loss_exp_alpha
            self.pos_weight = (1 - alpha) * self.pos_weight + alpha * self.get_batch_pos_weight(target)
        
        return torch.nn.functional.binary_cross_entropy_with_logits(
            output, target.to(output), pos_weight=self.pos_weight
        )
        
        
        
