from pathlib import Path
import torch
from torchvision.io import read_image

from dataset import SceneDataset
from utils.data_parsing import load_yaml_munch

cfg = load_yaml_munch(Path("utils") / "config.yaml")

def train_loader(image_names, camera_params_list, points, gt):
    images = torch.stack([read_image(image_name) for image_name in image_names]).to(cfg.device)
    transformation = torch.stack([torch.from_numpy(camera_params['K']@camera_params['T_cw'][:3, :]).float() for camera_params in camera_params_list]).to(cfg.device)
    points = torch.tensor(torch.from_numpy(points).float()).to(cfg.device)
    gt = torch.tensor(torch.from_numpy(gt).float()).to(cfg.device)
    return
    
    
if __name__ == "__main__":
    while True:
        dataset = SceneDataset(camera="iphone", n_points=300000, threshold_occ=0.01, representation="occ", visualize=False)
        data = dataset[0]
        points, gt = data['training_data']
        image_names, camera_params_list, _ = data['images']
        
        train_loader(image_names, camera_params_list, points, gt)