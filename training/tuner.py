from typing import Callable, Optional
from scipy.stats import distributions
from sklearn.model_selection import ParameterSampler
from lightning.pytorch import Trainer, LightningModule


learning_rate_space = [1e-4, 4e-4, 7e-4, 1e-3]

def tune(train_fn: Callable[[dict, dict, dict, str], tuple[Trainer, LightningModule]], default_config: dict, search_space: dict, num_samples: int, experiment_name: str, num_iterations = 4):

    param_grid = ParameterSampler(search_space, n_iter=num_samples)
    def do_run(params: dict):
        items = [f"{k}={v}" for k, v in params.items()]
        identifier = f"{experiment_name}_{'&'.join(items)}"

        trainer, module = train_fn(params, default_config, {}, identifier)

        metrics = trainer.callback_metrics
        # get all of the items, so that we are not having anything on the gpu anymore
        metrics = {k: v.item() for k, v in metrics.items()}

        del trainer, module

        return metrics, identifier

    results = {}

    configurations = [{ **config, "learning_rate": lr} for config in param_grid for lr in learning_rate_space]

    for i in range(num_iterations):
        for params in param_grid:
            for lr in learning_rate_space:
                params["learning_rate"] = lr
                metrics, identifier = do_run(params)
                results[identifier] = metrics
