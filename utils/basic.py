import torch


def get_default_device():
    """Pick GPU (cuda > mps) and else CPU"""
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
