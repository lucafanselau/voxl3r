import glob
import os

from lightning import Trainer

from experiments.mast3r_baseline.data import (
    Mast3rBaselineDataConfig,
    Mast3rBaselineDataModule,
)
from experiments.mast3r_baseline.module import (
    Mast3rBaselineConfig,
    Mast3rBaselineLightningModule,
)
from utils.data_parsing import load_yaml_munch

config = load_yaml_munch("./utils/config.yaml")


def main(args):

    # get last created folder in ./.lightning/surface-net-3d/surface-net-3d/
    # scenes=load_yaml_munch(Path("./data") / "dslr_undistort_config.yml").scene_ids,
    base_dir = config.data_dir
    pattern = os.path.join(
        base_dir, "*", "prepared_grids", "dslr", "*furthest_center_1.47"
    )
    matching_paths = glob.glob(pattern)

    # Extract the parent directories (two levels up from 'undistorted_images')
    scenes = [
        os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        for path in matching_paths
    ]

    # data_config = SurfaceNet3DDataConfig(data_dir=config.data_dir, batch_size=16, num_workers=11, scenes=load_yaml_munch(Path("./data") / "dslr_undistort_config.yml").scene_ids)
    data_config = Mast3rBaselineDataConfig(
        data_dir=config.data_dir,
        batch_size=16,
        num_workers=1,
        with_furthest_displacement=True,
        scenes=scenes,
        concatinate_pe=True,
    )

    datamodule = Mast3rBaselineDataModule(data_config=data_config)

    config = Mast3rBaselineConfig()

    model = Mast3rBaselineLightningModule(config=config)

    trainer = Trainer()

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
