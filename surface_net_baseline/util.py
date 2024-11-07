
from einops import rearrange
import torch
from jaxtyping import Float
from torch import Tensor


def project_points_to_images(points, images, transformations: Float[Tensor, ""]):
  """
  images: expected to be normalized [0, 1] float images C, H, W

  """


  # reshape points [num_points, 3] so that we have [num_points, num_images, 3]
  num_points = points.shape[0]
  num_images, _3, height, width = images.shape

  points = points.reshape(num_points, 1, 3).repeat(1, num_images, 1)
  # convert to homographic coordinates
  points = torch.cat([points, torch.full((num_points, num_images, 1), 1).to(cfg.device)], dim=-1)
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
  full_mask = mask.reshape(-1, 6, 1).repeat(1, 1, 2)
  valid_pixels = torch.where(full_mask, transformed_points, torch.nan)

  # now sample the image at the valid pixels
  # the way this is done currently all of the "invalid" uvs will be 0, but we only use the valid ones later so its fine
  grid = rearrange(valid_pixels, "p n two -> n 1 p two", two=2)
  # normalize in last dimension to range [-1, 1]
  grid = (grid / torch.tensor([width, height]).to(cfg.device)) * 2 - 1

  sampled = torch.nn.functional.grid_sample(images, grid)
  sampled = rearrange(sampled, "images channels 1 points -> points images channels")

  sampled[~mask] = fill_value

  # reshape to (num_points, num_images * 3)
  sampled = rearrange(sampled, "points images channels -> points (images channels)")