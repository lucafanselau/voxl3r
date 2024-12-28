

from datasets import transforms
from . import images
from . import occ
from . import mesh
from . import mast3r

class Config(mast3r.Config, occ.Config, images.Config, mesh.Config, transforms.SmearMast3rConfig):
    pass

class Visualizer(mast3r.Visualizer, images.Visualizer, occ.Visualizer, mesh.Visualizer):
    """
    Main visualizer class. Inherits from all other visualizer classes.
    """

    def __init__(self, config: Config):
        super().__init__(config)
