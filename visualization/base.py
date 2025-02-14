from datetime import datetime
import os
from typing import TypedDict
from pyvista import Plotter, global_theme
from jaxtyping import jaxtyped, Float
from torch import Tensor
from utils.config import BaseConfig


class Config(BaseConfig):
    log_dir: str = "logs"


Transformation = Float[Tensor, "4 4"]
BatchedTransformation = Float[Tensor, "B 4 4"]


class Visualizer:
    """
    Something that has a plotter
    """

    def __init__(self, config: Config):
        self.config = config
        self.plotter = self.create_plotter(config)

    def create_plotter(self, config: Config) -> Plotter:
        global_theme.allow_empty_mesh = True
        return Plotter()

    # TODO: add functions to show, screenshot, etc

    def save_screenshot(self, path: str) -> None:
        # make sure log_dir exists
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        path = os.path.join(self.config.log_dir, path)
        self.plotter.screenshot(path)

    def export_html_path(self, path: str):
        self.plotter.export_html(path)

    def export_html(self, path: str, timestamp: bool = False) -> None:
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)
        path = os.path.join(self.config.log_dir, path)
        if timestamp:
            path = f"{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.plotter.export_html(path + ".html")

    def export_gltf(self, path: str) -> None:
        self.plotter.export_gltf(path)

    def show(self) -> None:
        self.plotter.add_axes()
        self.plotter.enable_fly_to_right_click()
        # self.plotter.enable_joystick_style()
        self.plotter.enable_trackball_style()
        self.plotter.show()

    def add_axis(self, **kwargs) -> None:
        self.plotter.add_axes(**kwargs)
