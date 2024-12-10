

from . import images
from . import occ
from . import mesh
from . import mast3r

class Config(images.Config, occ.Config, mesh.Config, mast3r.Config):
    pass

class Visualizer(images.Visualizer, occ.Visualizer, mesh.Visualizer, mast3r.Visualizer):
    """
    Main visualizer class. Inherits from all other visualizer classes.
    """

    def __init__(self, config: Config):
        super().__init__(config)
