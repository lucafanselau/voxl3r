from abc import ABC, abstractmethod
import dataclasses
from beartype import beartype
import torch
from torch import Tensor
from jaxtyping import Float, jaxtyped

from baseline.mast3r_pipeline import Mast3rOutput

#
# Strategies for taking Mast3R Output and create a volume
#
#


# base implementation (as a stateful class)
class VolumeCreationStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_volume(self, mast3r_output: Mast3rOutput) -> Float[Tensor, "R R R F"]:
        pass


# "smeared" volume creation strategy
class SmearedVolumeCreationStrategy(VolumeCreationStrategy):
    def __init__(self):
        super().__init__()

    def create_volume(self, mast3r_output: Mast3rOutput) -> Float[Tensor, "R R R F"]:
        # create a volume from the predictions
        pass


def main():
    loaded_raw = torch.load("predictions.pt")
    loaded = Mast3rOutput.load("predictions.pt")

    # print all of the shapes of the values in the loaded object

    # create a volume from the predictions
    strategy = VolumeCreationStrategy()
    volume = strategy.create_volume(loaded)


if __name__ == "__main__":
    main()
