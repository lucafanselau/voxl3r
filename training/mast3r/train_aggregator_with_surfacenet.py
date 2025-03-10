"""
Example integration of SurfaceNet with the training pipeline.
This shows how to incorporate the SurfaceNet configuration into your existing training framework.
"""

from typing import Optional, Tuple, Union
from pydantic import Field
import torch
import wandb
from pytorch_lightning import Trainer

# Import your existing configs
from training.mast3r.train_aggregator import (
    DataConfig,
    LoggingConfig,
    TrainerConfig,
    UNet3DConfig,
    LightningModuleWithAuxConfig,
)

# Import SurfaceNet configuration
from networks.surfacenet import BaseSurfaceNetConfig, SurfaceNet

# =====================================
# Create a combined configuration class
# =====================================


class SurfaceNetTrainingConfig(
    LoggingConfig, TrainerConfig, DataConfig, BaseSurfaceNetConfig
):
    """Combined configuration for training SurfaceNet models"""

    resume: Union[bool, str] = False
    checkpoint_name: str = "last"

    # Add any additional configuration parameters specific to SurfaceNet training
    use_pretrained: bool = False
    pretrained_path: Optional[str] = None

    # Override defaults from BaseSurfaceNetConfig if needed
    size: str = "small"  # Default to small model


# =====================================
# Create a LightningModule for SurfaceNet
# =====================================


class SurfaceNetModule(torch.nn.Module):
    """Lightning module wrapper for SurfaceNet"""

    def __init__(self, config: SurfaceNetTrainingConfig):
        super().__init__()
        self.config = config
        self.model = SurfaceNet(config)

        # Define loss functions, optimizers, etc.
        # ...

    def forward(self, batch):
        return self.model(batch)

    # Add training_step, validation_step, etc.
    # ...


# =====================================
# Main training function
# =====================================


def train_surfacenet(
    config_path: str,
    size: str = "small",
    resume: bool = False,
):
    """
    Train a SurfaceNet model with the specified configuration

    Args:
        config_path: Path to the configuration YAML file
        size: Model size ("small", "medium", or "large")
        resume: Whether to resume training from a checkpoint
    """
    # Load the configuration
    config = SurfaceNetTrainingConfig.load_from_files([config_path])

    # Override the model size
    config.size = size

    # Set up the resume flag
    config.resume = resume

    # Create the model
    model = SurfaceNetModule(config)

    # Set up data module
    # datamodule = ...

    # Configure the trainer
    trainer_args = {
        "max_epochs": config.max_epochs,
        "precision": "bf16-mixed",
        "default_root_dir": "./.lightning/surfacenet",
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "num_sanity_val_steps": 0,
    }

    trainer = Trainer(
        **trainer_args,
        gradient_clip_val=2.0,
    )

    # Train the model
    # trainer.fit(model, datamodule=datamodule)

    # For demonstration purposes only
    print(f"Would train a {size} SurfaceNet model with:")
    print(f"  - in_channels: {config.in_channels}")
    print(f"  - l1_dim: {config.l1_dim}")
    print(f"  - l2_dim: {config.l2_dim}")
    print(f"  - l3_dim: {config.l3_dim}")
    print(f"  - l4_dim: {config.l4_dim}")
    print(f"  - l5_dim: {config.l5_dim}")
    print(f"  - side_dim: {config.side_dim}")


# =====================================
# Command-line interface
# =====================================


def main():
    """Command-line interface for training SurfaceNet models"""
    import argparse

    parser = argparse.ArgumentParser(description="Train SurfaceNet models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training")

    args = parser.parse_args()

    train_surfacenet(
        config_path=args.config,
        size=args.size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
