from typing import Optional
from einops import rearrange
from jaxtyping import jaxtyped, Bool, Float
import numpy as np
from sklearn.decomposition import PCA
from torch import Tensor

from utils.transformations import extract_rot_trans, invert_pose
from . import images
from . import base

import pyvista as pv

class Config(images.Config):
    pass

def apply_pca_to_image(image: Tensor, n_components=3) -> Tensor:
    # image shape expected: C H W
    C, H, W = image.shape
    
    # Flatten spatial dimensions, shape: (H*W, C)
    flattened = rearrange(image, "C H W -> (H W) C")
    
    # Fit and apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flattened.numpy())
    
    # Reshape back to H, W, n_components
    transformed_image = rearrange(transformed, "(H W) C -> H W C", H=H, W=W)
    
    # Convert back to Tensor if needed
    transformed_image = Tensor(transformed_image)
    return transformed_image

class Visualizer(images.Visualizer):
    def __init__(self, config: Config):
        super().__init__(config)
        
    def add_from_smearing_transform(self, smearing_dict: dict) -> None:
        debugging_dict = smearing_dict["verbose"]
        data_dict = debugging_dict["data_dict"]
        images = debugging_dict["images"]
        add_confidences = debugging_dict["add_confidences"]
        T_wc = debugging_dict["T_cw"]
        Ks = debugging_dict["K"]
        
        
        for i, (image, T_cw, K) in enumerate(zip(images, T_wc, Ks.values())):
            transformed_image = apply_pca_to_image(image)
            rearranged = rearrange(transformed_image, "C H W -> H W C")[:, :, :3].numpy()
            normalized = (rearranged - rearranged.min()) / (rearranged.max() - rearranged.min())
            scaled = normalized * 255.0
            result = scaled.astype(np.uint8)
            
            texture = pv.numpy_to_texture(result)
            
            _, _, transform =invert_pose(*extract_rot_trans(T_cw))
            self.add_image(texture, transform, K.numpy(), height=debugging_dict["height"], width=debugging_dict["width"], highlight=i == 0)
    

    def add_mast3r_images(self, 
                   images: Float[Tensor, "B H W F"], 
                   transformation: base.BatchedTransformation,
                   intrinsics: Float[Tensor, "B 3 3"],
                   confidences: Optional[Float[Tensor, "B H W 1"]] = None
                   ) -> None:
        """
        Add a batch of images to the visualizer.

        Args:
            images: HAS TO BE WITHOUT CONFIDENCES!
        """
        pass

