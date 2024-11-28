import os
import glob
import torch
from tqdm import tqdm

base_dir = '/home/luca/mnt/data/scannetpp/data'
pattern = os.path.join(base_dir, '*', 'prepared_grids', '**', '*.pt')
matching_paths = glob.glob(pattern, recursive=True)

def is_bool_3d_tensor(obj):
    return (
        isinstance(obj, torch.Tensor) and
        obj.dtype == torch.bool and
        obj.ndim == 4  and
        obj.shape[0] == 1
        
    )

# Extract the scene names
for chunk in tqdm(matching_paths):
    loaded_scene = torch.load(chunk)
    if len(loaded_scene["training_data"]) == 2 and is_bool_3d_tensor(loaded_scene["training_data"][1]):
        loaded_scene["training_data"] = loaded_scene["training_data"][1]
        torch.save(loaded_scene, chunk)
    elif is_bool_3d_tensor(loaded_scene["training_data"]):
        print(f"It seams like chunk {chunk} was already processed")
    else:
        print(f"Chunk {chunk} has an unexpected structure")
