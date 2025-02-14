from lightning import Trainer, Callback
from networks.attention_net import AttentionNet
from training.default.aux_module import LightningModuleWithAux, LightningModuleWithAuxConfig
from training.mast3r.train import Config as UNet3DConfig
from training.common import load_config_from_checkpoint, create_datamodule
from training.mast3r.train_attention import Config as TrainConfig



ConfigClass = TrainConfig
Config = ConfigClass
Module = LightningModuleWithAux

class LossCallback(Callback):
    def __init__(self):
        super().__init__()


    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):

        X, Y, images = batch["X"], batch["Y"], batch["images"]
        


        print(outputs)

def test_run(run_name, project_name):

    config, path = load_config_from_checkpoint(project_name, run_name, ConfigClass=ConfigClass)
    datamodule = create_datamodule(config, splits=["test"])

    module = Module.load_from_checkpoint(path, module_config=config, ModelClass=AttentionNet)

    callbacks = [LossCallback()]

    trainer = Trainer(max_epochs=1, callbacks=callbacks)
    trainer.test(module, datamodule)

if __name__ == "__main__":
    project_name = "mast3r-3d-experiments"
    run_name = "1tdd32ce"
    test_run(run_name, project_name)