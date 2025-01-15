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
from PIL import Image

class Config(images.Config):
    pass

def apply_pca_to_image(image: Tensor, n_components=3) -> Tensor:
    # image shape expected: C H W
    C, H, W = image.shape
    
    # Flatten spatial dimensions, shape: (H*W, C)
    flattened = rearrange(image, "C H W -> (H W) C")
    
    # Fit and apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flattened.numpy() if isinstance(flattened, Tensor) else flattened)
    
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
        T_cw_all = debugging_dict["T_cw"]
        
        image_names = [ele for pair in data_dict["pairs_image_names"] for ele in pair]
        Ks = [debugging_dict["K"][ele] for ele in image_names]
        
        
        for i, (image, T_cw, K) in enumerate(zip(images, T_cw_all, Ks)):
            if i%2 == 0:
                combined_image= np.concatenate(np.array(images[i:i+2]), axis=2)
                combined_image = combined_image[combined_image.reshape(24, -1).mean(axis=1).argsort()[:3]]
                transformed_image = apply_pca_to_image(combined_image)[:, :images[0].shape[2], :] #rearrange(image[:3,:,:], "C H W -> H W C") 
            else:
                combined_image= np.concatenate(np.array(images[i-1:i+1]), axis=2)
                combined_image = combined_image[combined_image.reshape(24, -1).mean(axis=1).argsort()[:3]]
                transformed_image = apply_pca_to_image(combined_image)[:, images[0].shape[2]:, :] #rearrange(image[:3,:,:], "C H W -> H W C") 
            normalized = (transformed_image - transformed_image.min()) / (transformed_image.max() - transformed_image.min())
            scaled = normalized * 255.0
            result = scaled.numpy().astype(np.uint8)
            
            #im = Image.fromarray(result)
            #im.save(f"./image_{i}.jpeg")
            
            texture = pv.numpy_to_texture(result)
            
            _, _, transform =invert_pose(*extract_rot_trans(T_cw))
            print(f"Image {i} is {image_names[i]}")
            self.add_image(texture, transform, K.numpy(), height=debugging_dict["height"], width=debugging_dict["width"], highlight=i)
            
    

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

