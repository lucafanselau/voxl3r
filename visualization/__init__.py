

# from datasets.transforms.smear_images import SmearMast3rConfig
from . import images
from . import occ
from . import mesh
from . import mast3r
from . import smeared_images

class Config(mast3r.Config, occ.Config, images.Config, mesh.Config, smeared_images.Config):
    pass

class Visualizer(mast3r.Visualizer, images.Visualizer, occ.Visualizer, mesh.Visualizer, smeared_images.Visualizer):
    """
    Main visualizer class. Inherits from all other visualizer classes.
    """

    def __init__(self, config: Config):
        super().__init__(config)
