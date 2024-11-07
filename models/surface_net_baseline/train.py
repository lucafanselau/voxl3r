from pathlib import Path
import torch
from torchvision.io import read_image

from dataset import SceneDataset
from einops import rearrange
from models.surface_net_baseline.model import SimpleOccNetConfig
from models.surface_net_baseline.module import LRConfig, OccSurfaceNet, OptimizerConfig
from models.surface_net_baseline.util import project_points_to_images
from utils.data_parsing import load_yaml_munch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import lightning as pl

from utils.visualize import visualize_mesh


cfg = load_yaml_munch(Path("utils") / "config.yaml")

def train_loader(image_names, camera_params_list, points, gt):
    images = torch.stack([read_image(image_name) for image_name in image_names]).to(cfg.device)
    transformation = torch.stack([torch.from_numpy(camera_params['K']@camera_params['T_cw'][:3, :]).float() for camera_params in camera_params_list]).to(cfg.device)
    points = torch.tensor(torch.from_numpy(points).float()).to(cfg.device)
    gt = torch.tensor(torch.from_numpy(gt).float()).to(cfg.device)
    return images, transformation, points, gt
    
    
if __name__ == "__main__":
    dataset = SceneDataset(data_dir="data", camera="iphone", n_points=300000, threshold_occ=0.01, representation="occ", visualize=False)
    idx = dataset.get_index_from_scene("8b2c0938d6")
    data = dataset[idx]
    points, gt = data['training_data']
    image_names, camera_params_list, _ = data['images']
    
    images, transformations, points, gt = train_loader(image_names, camera_params_list, points, gt)
    # and normalize images
    images = images / 255.0
    
    X = project_points_to_images(points, images, transformations)
    
    rgb_list = rearrange(X, 'p (i c) -> p i c', c=3)
    mask = rgb_list != -1
    denom = torch.sum(torch.sum(mask, -1)/3, -1)
    rgb_list[rgb_list == -1.0] = 0.0
    rgb_list_pruned = rgb_list[denom != 0]
    points = points[denom != 0]
    denom = denom[denom != 0]
    rgb_list_avg = torch.sum(rgb_list_pruned, dim=1) / denom.unsqueeze(-1).repeat(1, 3)

    visualize_mesh(None, point_coords=points.cpu().numpy(), heat_values=gt.cpu().numpy(), rgb_list=rgb_list_avg.cpu().numpy())
    # target values will be gt
    target = rearrange(gt, "points -> points 1")
    
    
    

    # Dataloader for [X, target] Tensor
    # split the data
    dataset = torch.utils.data.TensorDataset(X, target)
    train, val, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

    # create dataloaders
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    # Initialize OccSurfaceNet
    model = OccSurfaceNet(SimpleOccNetConfig(input_dim=images.shape[0]*3, hidden=[512,512,512]),  OptimizerConfig(), LRConfig())
    logger = WandbLogger(project="surface-net-baseline", save_dir="./.lightning/occ-surface-net")

    trainer = Trainer(
        max_epochs=1000,
        # Used to limit the number of batches for testing and initial overfitting
        #limit_train_batches=8,
        #limit_val_batches=2,
        # Logging stuff
        #log_every_n_steps=2,
        logger=logger,
        profiler="simple",
        # Performance stuff
        precision="bf16-mixed",
        default_root_dir="./.lightning/occ-surface-net",
    )
    trainer.fit(model, train_loader, val_loader)
