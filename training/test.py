from lightning import Trainer, Callback
from training.mast3r.module_unet3d import UNet3DLightningModule
from training.mast3r.train import Config as UNet3DConfig
from training.common import load_config_from_checkpoint, create_datamodule


ConfigClass = UNet3DConfig

class LossCallback(Callback):
    def __init__(self):
        super().__init__()


    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):

        X, Y, images = batch["X"], batch["Y"], batch["images"]
        


        print(outputs)

def test_run(run_name, project_name):

    config, path = load_config_from_checkpoint(project_name, run_name, ConfigClass=ConfigClass)
    datamodule = create_datamodule(config, splits=["test"])

    module = UNet3DLightningModule.load_from_checkpoint(path, module_config=config)

    callbacks = [LossCallback()]

    trainer = Trainer(max_epochs=1, callbacks=callbacks)
    trainer.test(module, datamodule)

if __name__ == "__main__":
    project_name = "mast3r-3d-experiments"
    run_name = "NxY0Y_0_0"
    test_run(run_name, project_name)