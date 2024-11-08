
from einops import rearrange
import torch
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
import lightning as pl

from dataset import SceneDataset, SceneDatasetTransformToTorch


@jaxtyped(typechecker=beartype)
def project_points_to_images(points: Float[Tensor, "Points 3"], images: Float[Tensor, "Images 3 H W"], transformations: Float[Tensor, "Images 3 4"]) -> Float[Tensor, "Points Images*3"]:
  """
  images: expected to be normalized [0, 1] float images C, H, W

  """
  # reshape points [num_points, 3] so that we have [num_points, num_images, 3]
  num_points = points.shape[0]
  num_images, _3, height, width = images.shape

  points = points.reshape(num_points, 1, 3).repeat(1, num_images, 1)
  # convert to homographic coordinates
  points = torch.cat([points, torch.full((num_points, num_images, 1), 1).to(points)], dim=-1)
  # and convert it to "matrix" (4x1) (num_points, num_images, 4, 1)
  points = rearrange(points, 'n i d -> n i d 1')

  # perform the matmul with broadcasting (num_points, num_images, 4, 4) x (num_points, num_images, 4, 1) -> (num_points, num_images, 3, 1)
  transformed_points = torch.matmul(transformations, points)

  # remove last dimension (num_points, num_images, 3)
  transformed_points = rearrange(transformed_points, 'n i d 1 -> n i d')

  # perform perspective division
  transformed_points = transformed_points[:, :, :3] / transformed_points[:, :, 2].unsqueeze(-1)
  # and drop the last dimension
  transformed_points = transformed_points[:, :, :2]

  # now we need to mask by the bounding box of the image and replace with a constant fill (-inf)
  fill_value = -1.0
  mask = (transformed_points[:, :, 0] >= 0) & (transformed_points[:, :, 0] < width) & (transformed_points[:, :, 1] >= 0) & (transformed_points[:, :, 1] < height)
  full_mask = mask.reshape(-1, num_images, 1).repeat(1, 1, 2)
  valid_pixels = torch.where(full_mask, transformed_points, torch.nan)

  # now sample the image at the valid pixels
  # the way this is done currently all of the "invalid" uvs will be 0, but we only use the valid ones later so its fine
  grid = rearrange(valid_pixels, "p n two -> n 1 p two", two=2)
  # normalize in last dimension to range [-1, 1]
  grid = (grid / torch.tensor([width, height]).to(grid)) * 2 - 1

  sampled = torch.nn.functional.grid_sample(images, grid)
  sampled = rearrange(sampled, "images channels 1 points -> points images channels")

  sampled[~mask] = fill_value

  # reshape to (num_points, num_images * 3)
  sampled = rearrange(sampled, "points images channels -> points (images channels)")

  return sampled

class OccSurfaceNetDatamodule(pl.LightningDataModule):
  def __init__(self, dataset: SceneDataset, scene_id: str, batch_size = 512, max_sequence_length = 20, target_device="mps"):
    super().__init__()
    self.dataset = dataset
    self.scene_id = scene_id
    self.scene_idx = dataset.get_index_from_scene(scene_id)
    self.transform = SceneDatasetTransformToTorch(target_device)
    self.batch_size = batch_size
    self.max_sequence_length = max_sequence_length
    self.parsed = None

  def prepare_data(self):
    scene_dataset = self.dataset
    idx = scene_dataset.get_index_from_scene(self.scene_id)
    data = scene_dataset[idx]
    mesh = data['mesh']
    points, gt = data['training_data']
    image_names, camera_params_list, _ = data['images']

   
    images, transformations, points, gt = self.transform.forward(data) 
    # and normalize images
    images = images / 255.0
    
    X = project_points_to_images(points, images, transformations)

    # we need to pad X to contain 3 * self.max_sequence_length elements in the last dimension (fill_value is -1)
    fill_value = -1.0
    num_points, num_images_flattened = X.shape
    X = torch.nn.functional.pad(X, (0, 3 * self.max_sequence_length - num_images_flattened), value=fill_value)
    
    # target values will be gt
    target = rearrange(gt, "points -> points 1")

    # Dataloader for [X, target] Tensor
    # split the data
    dataset = torch.utils.data.TensorDataset(X, target, points)

    return dataset

  def setup(self, stage):
    if self.parsed is None:
      self.parsed = self.prepare_data()
      generator = torch.Generator().manual_seed(42)
      self.split = torch.utils.data.random_split(self.parsed, [0.6, 0.2, 0.2], generator=generator)

    if stage == 'fit' or stage is None:
      self.train_dataset = self.split[0]
      self.val_dataset = self.split[1]
      

    if stage == 'test' or stage is None:
      self.test_dataset = self.split[2]

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
  
  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
  
  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
  
  # def predict_dataloader(self):
  #   return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

