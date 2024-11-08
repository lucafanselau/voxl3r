from pathlib import Path
import numpy as np
import torch
from torchvision.io import read_image

from dataset import SceneDataset, SceneDatasetTransformToTorch
from einops import rearrange
from models.surface_net_baseline.model import SimpleOccNetConfig
from models.surface_net_baseline.module import LRConfig, OccSurfaceNet, OptimizerConfig
from models.surface_net_baseline.util import project_points_to_images
from utils.data_parsing import load_yaml_munch
from lightning import Trainer 
from lightning.pytorch.loggers import WandbLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint 

from utils.visualize import plot_voxel_grid, visualize_mesh


cfg = load_yaml_munch(Path("utils") / "config.yaml")
visualize = False

if __name__ == "__main__":
    dataset = SceneDataset(
        data_dir="datasets/scannetpp/data",
        camera="iphone",
        n_points=300000,
        threshold_occ=0.01,
        representation="occ",
        visualize=False,
    )
    idx = dataset.get_index_from_scene("8b2c0938d6")
    data = dataset[idx]
    transform = SceneDatasetTransformToTorch(cfg.device)

    image_names, camera_params_list, _ = data["images"]
    images, transformations, points, gt = transform.forward(data)
    # and normalize images
    images = images / 255.0

    X = project_points_to_images(points, images, transformations)
    
    rgb_list = rearrange(X, 'p (i c) -> p i c', c=3)
    mask = rgb_list != -1
    denom = torch.sum(torch.sum(mask, -1)/3, -1)
    rgb_list[rgb_list == -1.0] = 0.0
    rgb_list_pruned = rgb_list[denom != 0]
    points_pruned = points[denom != 0]
    occ = gt[denom != 0]
    denom = denom[denom != 0]
    rgb_list_avg = torch.sum(rgb_list_pruned, dim=1) / denom.unsqueeze(-1).repeat(1, 3)

    visualize_mesh(mesh, point_coords=points_pruned.cpu().numpy(), images=image_names, camera_params_list=camera_params_list,  heat_values=occ.cpu().numpy(), rgb_list=rgb_list_avg.cpu().numpy())
    
if __name__ == "__main__":
    np.random.seed(42)
    
    scene_dataset = SceneDataset(data_dir="data", camera="iphone", n_points=300000, threshold_occ=0.01, representation="occ", visualize=False)
    
    visualize_unprojection(scene_dataset, scene="8b2c0938d6")

    idx = scene_dataset.get_index_from_scene("8b2c0938d6")
    data = scene_dataset[idx]
    mesh = data['mesh']
    points, gt = data['training_data']
    image_names, camera_params_list, _ = data['images']
    
    images, transformations, points, gt = train_loader(image_names, camera_params_list, points, gt)
    # and normalize images
    images = images / 255.0
    
    X = project_points_to_images(points, images, transformations)

    idx = scene_dataset.get_index_from_scene("8b2c0938d6")
    
    points, gt = scene_dataset.get_all_data_scene(idx)
    
    X = project_points_to_images(points, images, transformations)
    
    # target values will be gt
    target = rearrange(gt, "points -> points 1")

    # Dataloader for [X, target] Tensor
    # split the data
    dataset = torch.utils.data.TensorDataset(X, target, points)
    generator = torch.Generator().manual_seed(42)
    train, val, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    # create dataloaders
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    model = OccSurfaceNet.load_from_checkpoint(".lightning/occ-surface-net/surface-net-baseline/wjcst3w3/checkpoints/epoch=340-step=8866.ckpt")
    
    # Initialize OccSurfaceNet
    model = OccSurfaceNet(
        SimpleOccNetConfig(input_dim=images.shape[0] * 3, hidden=[512, 512, 512]),
        OptimizerConfig(),
        LRConfig(),
    )
    logger = WandbLogger(
        project="surface-net-baseline", save_dir="./.lightning/occ-surface-net"
    )

    # custom ModelCheckpoint
    # Save top3 models wrt precision
    filename = "{epoch}-{step:.2f}"
    callbacks = [ModelCheckpoint(
        filename=filename + "-{i}",
        monitor=f"val_{i}",
        save_top_k=3,
        mode=mode,
    ) for [i, mode] in [["accuracy", "max"], ["loss", "min"]]]
    
    # Save the model every 5 epochs
    every_five_epochs = ModelCheckpoint(
        period=5,
        save_top_k=-1,
        save_last=True,
    )



    trainer = Trainer(
        max_epochs=400,
        # Used to limit the number of batches for testing and initial overfitting
        # limit_train_batches=8,
        # limit_val_batches=2,
        # Logging stuff
        # log_every_n_steps=2,
        callbacks=[*callbacks, every_five_epochs],
        logger=logger,
        profiler="simple",
        # Performance stuff
        precision="bf16-mixed",
        default_root_dir="./.lightning/occ-surface-net",
    )
    #trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
    gt = torch.cat(model.test_record['gt'])
    points = torch.cat(model.test_record['points'])
    y = torch.sigmoid(torch.cat(model.test_record['out']))
    y[y < 0.5] = 0.0
    y[y > 0.5] = 1.0
    
    mesh = Path(scene_dataset.data_dir) / scene_dataset.scenes[idx] / "scans" / "mesh_aligned_0.05.ply"
    #visualize_mesh(mesh, point_coords=points.cpu().numpy(), heat_values=y.cpu().numpy())
    plot_voxel_grid(points.cpu().numpy(), y.cpu().numpy())
    
    
    trainer.fit(model, train_loader, val_loader)

    print("Running test on best model...")
    # this should be with regard to the validation set
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=test_loader)
