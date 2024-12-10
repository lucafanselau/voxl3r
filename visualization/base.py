from typing import TypedDict
from pyvista import Plotter
from jaxtyping import jaxtyped, Float
from torch import Tensor

class Config(TypedDict):
    pass


Transformation = Float[Tensor, "4 4"]
BatchedTransformation = Float[Tensor, "B 4 4"]

class Visualizer:
    """
    Something that has a plotter

    """

    def __init__(self, config: Config):
        self.plotter = self.create_plotter(config)

    def create_plotter(self, config: Config) -> Plotter:
        pass

    # TODO: add functions to show, screenshot, etc

    def save_screenshot(self, path: str) -> None:
        pass

    def show(self) -> None:
        pass

