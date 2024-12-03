from torch import nn
import torch
from dataclasses import dataclass, field

@dataclass
class SimpleOccNetConfig:
  input_dim: int = 18
  hidden: list[int] = field(default_factory=lambda: [64, 64, 64])
  output_dim: int = 1

class SimpleOccNet(nn.Module):
  """
  Simple feedforward fully-connected network, with ReLU activation functions
  """

  def __init__(self, cfg: SimpleOccNetConfig):
    super(SimpleOccNet, self).__init__()

    # first concatenate hidden with input and output
    hidden = [cfg.input_dim] + cfg.hidden
    self.blocks = nn.ModuleList([nn.Sequential(
      nn.Linear(hidden[i], hidden[i + 1]),
      nn.ReLU()
    ) for i in range(len(hidden) - 1)])

    self.blocks.append(nn.Linear(hidden[-1], cfg.output_dim))

  def forward(self, x):
    for block in self.blocks:
      x = block(x)

    # output activation sigmoid
    return x
