import math
import random
import string
from typing import Callable, List, Optional
from loguru import logger
from scipy.stats import distributions
from sklearn.model_selection import ParameterSampler
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import LearningRateFinder
import torch
import wandb


default_lr_search_space = [1e-3]#[1e-3]#[7e-4, 1e-3, 4e-3]
#default_lr_search_space = [7e-4, 1e-3]

def _tune_impl(
        train_fn: Callable[[dict, dict, dict, str, str], tuple[Trainer, LightningModule]], 
        default_config: dict, 
        search_space: dict, 
        num_samples: int, 
        experiment_name: str,
        learning_rate_space: List[float],
        monitor: str,
        mode: str,
        base_epochs: int,
        final_epochs: int,
        dry_run: bool = False
        
    ):

    def do_run(params: dict, identifier: str, specific_run_name: str):
        logger.info(f"Starting training for {identifier} ({specific_run_name})")
        if dry_run:
            return { monitor: 0.0 }, identifier

        trainer, module = train_fn(params, default_config, {}, identifier, specific_run_name, experiment_name)
        logger.info(f"Training finished for {identifier} ({specific_run_name})")
        metrics = trainer.callback_metrics
        # get all of the items, so that we are not having anything on the gpu anymore
        metrics = {k: v.item() for k, v in metrics.items()}

        del trainer, module

        return metrics, identifier
    
    # prepare some config stuff
    default_config.skip_prepare = True

    # random string for the run name
    run_name = "".join(random.choices(string.ascii_letters + string.digits, k=5))

    param_grid = ParameterSampler(search_space, n_iter=num_samples)
    results = {}
    param_grid_index = 0
    for params in param_grid:
        # we do this for every param
        # we want to run the first base_epochs epochs with all of the learning rates
        # after that we want to cut the number of learning rates in half (k = n/2) and only run the best k learning rates repeat the process until we have 1 learning rate left

        logger.info(f"Running for parameters {params}")
        current_learning_rate_space = [*learning_rate_space]

        # maximum number of refinement steps
        num_refinement_steps = math.floor(math.log2(len(learning_rate_space))) + 1

        # Each single run should be 15 epochs
        # last refinement step should be base_epochs + 30 epochs
        default_config.max_epochs = base_epochs * num_refinement_steps + final_epochs
        # we want
        for i in range(num_refinement_steps):
            logger.info(f"Refinement step {i + 1} of {num_refinement_steps}")
            # this is what actually limits the number of epochs in the lightning trainer
            default_config.limit_epochs = base_epochs * (i + 1) if i < num_refinement_steps - 1 else base_epochs * num_refinement_steps + final_epochs
            stage_results: list[tuple[float, float]] = []
            for lr in current_learning_rate_space:
                params["learning_rate"] = lr
                try:

                    items = [f"{k}={v}" for k, v in params.items()]
                    specific_run_name = f"{run_name}_{'&'.join(items)}"
                    identifier = f"{run_name}_{param_grid_index}_{default_lr_search_space.index(lr)}"

                    if identifier in results and results[identifier] == "failed":
                        # in case we already know that this run failed, we can skip it
                        logger.warning(f"Skipping {identifier} because it failed previously")
                        continue

                    metrics, identifier = do_run(params, identifier, specific_run_name)

                    # This is for the actual output of the function
                    results[identifier] = metrics
                    stage_results.append((metrics[monitor], lr))
                except Exception as e:
                    logger.error(f"Error tuning: Params: {params} Learning Rate: {lr} Error: {e}")
                    # store the worst possible value
                    stage_results.append((float('inf') if mode == "min" else -float('inf'), lr))
                    # signal in results that this run failed
                    results[identifier] = "failed"
                    # signal in the trainer that this run failed
                    raise e
                finally:
                    wandb.finish(quiet=True)
                    del metrics, identifier
            
            # We want to keep the best k learning rates
            stage_results.sort(key=lambda x: x[0], reverse=mode == "max")
            current_learning_rate_space = [x[1] for x in stage_results[:math.floor(len(stage_results) / 2)]]

        param_grid_index += 1

    return results

def tune(
        train_fn: Callable[[dict, dict, dict, str, str], tuple[Trainer, LightningModule]], 
        default_config: dict,
        search_space: dict, 
        num_samples: int, 
        experiment_name: str,
        learning_rate_space: List[float] = default_lr_search_space,
        monitor: str = "val_loss",
        mode: str = "min",
        base_epochs: int = 50,
        final_epochs: int = 100,
        dry_run: bool = False
    ):
    results = _tune_impl(train_fn, default_config, search_space, num_samples, experiment_name, learning_rate_space, monitor, mode, base_epochs, final_epochs, dry_run)

    import datetime
    import pathlib
    save_path = pathlib.Path(f"./experiments/{experiment_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth")

    return results
