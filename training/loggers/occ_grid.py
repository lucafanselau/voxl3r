import io
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple
from einops import rearrange
from lightning import Callback, LightningModule, Trainer
import numpy as np
from torch import Tensor
from lightning.pytorch.loggers import WandbLogger
import torch
from jaxtyping import Float
from loguru import logger
import tqdm
import trimesh
import wandb
import heapq
import threading
from queue import Queue, Empty, Full
import time
from visualization.occ import Visualizer as OccVisualizer
from visualization.images import Visualizer as ImageVisualizer
from visualization.base import Config as VisConfig
from utils.transformations import invert_pose, extract_rot_trans


# Pytorch Lightning Callback to log the 3D voxel grids at end of epoch
class OccGridCallback(Callback):
    # we also store the batch idx to know which batch is which
    # Tuple is [pred, batch_idx, gt, dataset_idx]
    results = {
        "train": {"best": [], "worst": []},
        "val": {"best": [], "worst": []},
        "test": {"best": [], "worst": []},
    }

    def __init__(
        self,
        wandb: WandbLogger,
        n_epochs: Tuple[int, int, int] = (5, 5, 1),
        max_results: int = 5,
        config: VisConfig = VisConfig(),
    ):
        super().__init__()
        self.worker = VoxelGridWorker(
            wandb, config, max_results
        )  # Replace direct wandb reference
        self.n_epochs = {"train": n_epochs[0], "val": n_epochs[1], "test": n_epochs[2]}
        self.max_results = max_results

    # use the on_batch_end to store the first result of the epoch
    def sink_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        type: str,
    ) -> None:
        # outputs is a dict with the loss and the predictions
        for i, loss in enumerate(outputs["loss_batch"]):

            logger = (
                {
                    key: (
                        value[i].detach().cpu().numpy()
                        if not isinstance(value, list)
                        else [value[j][i] for j in range(len(value))]
                    )
                    for key, value in batch["logger"].items()
                }
                if "logger" in batch.keys()
                else None
            )

            def add_result(best_or_worst: str, loss, j, pair_idx):
                if len(self.results[type][best_or_worst]) < self.max_results:
                    method = heapq.heappush
                else:
                    method = heapq.heappushpop

                if (outputs["pred"][i][j] > 0.5).sum() > outputs["pred"][i][
                    j
                ].numel() * 0.3:
                    method(
                        self.results[type][best_or_worst],
                        (
                            -loss[j] if best_or_worst == "best" else loss[j],
                            {
                                "pred": torch.sigmoid(outputs["pred"][i][j])
                                .detach()
                                .cpu(),
                                "gt": batch["Y"][i].detach().cpu(),
                                "logger": logger,
                                "pair_idx": pair_idx,
                            },
                        ),
                    )

            # TWO cases:
            # loss batches is (B, 1) -> eg. just a single prediction, eg. all of the images are considered
            # loss batches is (B, N) -> eg. one prediction per image, eg. only a subset of the images are considered
            if loss.numel() == 1:
                # -> just a single prediction
                j = 0
                add_result("worst", loss, j, None)
                add_result("best", loss, j, None)
            else:
                j = torch.argmax(loss).item()
                add_result("worst", loss, j, j)

                # now we also need to add the best
                j = torch.argmin(loss).item()
                add_result("best", loss, j, j)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        kwargs["type"] = "train"
        try:
            self.sink_on_batch_end(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[OccGridCallback] Error storing voxel grid: {e}")

    def on_validation_batch_end(self, *args, **kwargs) -> None:
        kwargs["type"] = "val"
        try:
            self.sink_on_batch_end(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[OccGridCallback] Error storing voxel grid: {e}")

    def on_test_batch_end(self, *args, **kwargs) -> None:
        kwargs["type"] = "test"
        try:
            self.sink_on_batch_end(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[OccGridCallback] Error storing voxel grid: {e}")

    def create_voxel_grid(
        self,
        grid: Float[Tensor, "1 X Y Z"],
        occ_threshold: float = 0.5,
        idx: int = 0,
    ) -> wandb.Object3D:
        # grid is a tensor of shape (1, X, Y, Z)

        # create a trimesh object for this grid
        occ = grid
        _1, X, Y, Z = occ.shape

        encoding = trimesh.voxel.encoding.DenseEncoding(
            (occ > occ_threshold).detach().squeeze(0).bool().cpu().numpy()
        )

        grid = trimesh.voxel.VoxelGrid(encoding)
        occ_re = rearrange(occ, "1 X Y Z -> X Y Z 1")
        # visualize the occ as a opacity
        colors = (
            ((occ_re.float() - occ_threshold) / (1 - occ_threshold))
            .repeat(1, 1, 1, 4)
            .float()
        )
        # colors[..., 3] = 1
        if (occ > 0.5).sum() == 0:
            return None, None
        mesh = grid.as_boxes(colors=colors.detach().cpu().numpy())

        tmp_dir = Path(self.wandb.experiment.dir) / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        mesh.export(str(tmp_dir / f"voxel_grid_{idx}.glb"))
        object_3d = wandb.Object3D(str(tmp_dir / f"voxel_grid_{idx}.glb"))
        # delete the tmp file
        file_name = tmp_dir / f"voxel_grid_{idx}.glb"
        return object_3d, file_name

    def log_results(self, trainer: Trainer, type: str):
        """Push results to worker queue instead of processing directly"""
        results = self.results[type]

        with tqdm.tqdm(
            total=len(results["best"]) + len(results["worst"]), desc="Visualizing"
        ) as pbar:
            for key in ["best", "worst"]:
                entries = results[key]
                for i in range(len(entries)):
                    try:
                        pred = entries[i][1]["pred"]
                        gt = entries[i][1]["gt"]

                        # Non-blocking put with timeout
                        self.worker._process_task(
                            {
                                "type": type,
                                "key": key,
                                "pred": pred,
                                "gt": gt,
                                "idx": i,
                                "logger": entries[i][1][
                                    "logger"
                                ],  # Pass the original batch data
                                "pair_idx": entries[i][1]["pair_idx"],
                            },
                        )
                        pbar.update(1)
                    except Full:
                        logger.warning(
                            "Voxel queue full, dropping some grid logging tasks"
                        )
                    except Exception as e:
                        logger.error(f"Failed to queue voxel task: {str(e)}")

    # For now this is down only at the end of the training epoch
    def on_train_epoch_end(self, trainer, pl_module):
        if self.results["train"] is None:
            return
        if (trainer.current_epoch + 1) % self.n_epochs["train"] == 0:
            try:
                self.log_results(trainer, "train")
            except Exception as e:
                logger.warning(f"[OccGridCallback] Error logging voxel grid: {e}")
        # reset the train results
        self.results["train"] = {"best": [], "worst": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.results["val"] is None:
            return
        if (trainer.current_epoch + 1) % self.n_epochs["val"] == 0:
            try:
                self.log_results(trainer, "val")
            except Exception as e:
                logger.warning(f"[OccGridCallback] Error logging voxel grid: {e}")
        # reset the val results
        self.results["val"] = {"best": [], "worst": []}

    def on_test_epoch_end(self, trainer, pl_module):
        if self.results["test"] is None:
            return
        if (trainer.current_epoch + 1) % self.n_epochs["test"] == 0:
            try:
                self.log_results(trainer, "test")
            except Exception as e:
                logger.warning(f"[OccGridCallback] Error logging voxel grid: {e}")
        # reset the test results
        self.results["test"] = {"best": [], "worst": []}

    # Add cleanup method
    # def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     self.worker.shutdown()

    # def __del__(self):
    #     self.worker.shutdown()


class OccGridCallbackVisualizer(OccVisualizer, ImageVisualizer):
    def __init__(self, config: VisConfig):
        super().__init__(config)


class VoxelGridWorker:
    """Background thread for processing voxel grid generation and logging"""

    def __init__(self, wandb_logger: WandbLogger, config: VisConfig, max_results: int):
        self.queue = Queue(maxsize=max_results * 2)  # Prevent unbounded memory growth
        self.wandb = wandb_logger
        self.config = config
        self._shutdown_flag = threading.Event()
        # self.thread = threading.Thread(
        #     target=self._run_loop,
        #     name="VoxelGridWorker",
        #     daemon=True,  # Ensures thread exits with main process
        # )
        # self.thread.start()
        # self.visualizer: OccGridCallbackVisualizer | None = None
        logger.info("Started voxel grid worker thread")

    def _run_loop(self):
        """Main processing loop for the worker thread"""
        while self.queue.qsize() > 0 or not self._shutdown_flag.is_set():
            try:
                task = self.queue.get(timeout=2)  # Allows periodic shutdown check
                self._process_task(task)
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Voxel worker error: {str(e)}")
                time.sleep(1)  # Prevent tight error loops

    def _log_only_voxel_grid(self, grid: Tensor, idx: int, label: str):
        """Log only the voxel grid"""
        try:
            obj_pred, file_pred = self._create_voxel_grid(grid, idx)
            self.wandb.experiment.log(
                {
                    f"{label}": obj_pred,
                }
            )
        finally:
            if file_pred and file_pred.exists():
                try:
                    file_pred.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_pred}: {str(e)}")

    def _setup_visualizer(self):
        """Setup the visualizer"""
        if self.visualizer is None:
            self.visualizer = OccGridCallbackVisualizer(self.config)

    def _log_voxel_grid_and_images(
        self,
        grid: (
            Float[Tensor, "1 X Y Z"]
            | Tuple[Float[Tensor, "1 X Y Z"], Float[Tensor, "3 X Y Z"]]
        ),
        idx: int,
        label: str,
        extrinsics_cw: Float[Tensor, "B 4 4"],
        K: Float[Tensor, "B 3 3"],
        origin: Float[Tensor, "3"],
        resolution: float,
        images: List[str],
        height: int,
        width: int,
        type: Literal["diff", "grid"] = "grid",
    ):
        """Log the voxel grid and the images"""

        try:

            path = None

            visualizer = OccGridCallbackVisualizer(self.config)

            import pyvista as pv

            _, _, T_world_object = invert_pose(*extract_rot_trans(extrinsics_cw[0]))

            if type == "diff":
                (gt, pred) = grid
                # gt is bool, pred is float [0, 1]
                diff = gt.float() - pred

                # we want to show

                # grid is the mask (eg. which voxels to even show)
                # we show voxels that are in the gt or would be classified as occupied by the pred
                mask = (gt | (pred > 0.5)).int()
                grid = mask

                color = torch.zeros((4, *mask.shape[1:]))

                # all opacity 1 for now
                color[3, ...] = 1

                # positive diff is red (eg. we have a gt voxel)
                # negative diff is green (eg. we have a pred voxel)
                color[0, (diff > 0).squeeze(0)] = diff[diff > 0]
                color[1, (diff < 0).squeeze(0)] = diff[diff < 0].abs()

                # visualize common set as full opacity white voxel grid
                pred_occ = pred > 0.5
                common = gt & pred_occ
                visualizer.add_occupancy(
                    occupancy=common,
                    T_world_object=torch.tensor(T_world_object),
                    origin=origin,
                    pitch=resolution.item(),
                    color=torch.ones_like(gt).repeat(3, 1, 1, 1).float(),
                    opacity=1.0,
                    # color=torch.ones_like(gt).repeat(4, 1, 1, 1).float(),
                )

                # next up visualize those that are in gt but not in pred
                gt_missing = gt & ~pred_occ
                # these we visualize as red, where the opacity is the inverse of the pred
                # eg. predicting 0.5 -> 0 opacity
                # predicting 0 -> 1 opacity
                gt_missing_color = torch.zeros_like(gt).repeat(4, 1, 1, 1).float()
                gt_missing_color[0, ...] = 1
                gt_missing_color[3, ...] = (0.5 - pred) * 2
                visualizer.add_occupancy(
                    occupancy=gt_missing,
                    T_world_object=torch.tensor(T_world_object),
                    origin=origin,
                    pitch=resolution.item(),
                    color=gt_missing_color,
                )

                # next up visualize those that are in pred but not in gt
                pred_missing = ~gt & pred_occ
                # these we visualize as red, where the opacity is the inverse of the pred
                # eg. predicting 1 -> 1 opacity
                # predicting 0.5 -> 0 opacity
                pred_missing_color = torch.zeros_like(gt).repeat(4, 1, 1, 1).float()
                pred_missing_color[1, ...] = 1
                pred_missing_color[3, ...] = (pred * 2) - 1
                visualizer.add_occupancy(
                    occupancy=pred_missing,
                    T_world_object=torch.tensor(T_world_object),
                    origin=origin,
                    pitch=resolution.item(),
                    color=pred_missing_color,
                )

            else:
                grid = grid

                visualizer.add_occupancy(
                    occupancy=grid,
                    T_world_object=torch.tensor(T_world_object),
                    origin=origin,
                    pitch=resolution.item(),
                )

            for i, (image, extrinsic_cw, intrinsic) in enumerate(
                zip(images, extrinsics_cw, K)
            ):
                texture = pv.read_texture(image)
                _, _, T_wc = invert_pose(*extract_rot_trans(extrinsic_cw))
                transform = T_wc
                visualizer.add_image(
                    texture,
                    transform,
                    np.asarray(intrinsic),
                    height,
                    width,
                    highlight=i,
                )

            tmp_dir = Path(self.wandb.experiment.dir) / "tmp"
            tmp_dir.mkdir(exist_ok=True)

            path = tmp_dir / f"voxel_grid_{idx}.gltf"
            visualizer.export_gltf(str(path))
            # visualizer.export_html("occ_grid_debug")
            self.wandb.experiment.log({f"{label}": wandb.Object3D(str(path))})
        finally:
            if path and path.exists():
                try:
                    path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up {path}: {str(e)}")

    def _process_task(self, task: dict):
        """Handle one logging task with proper cleanup"""
        mode = "diff"
        try:
            # Unpack task data
            grid_pred = task["pred"]
            grid_gt = task["gt"]
            log_type = task["type"]
            key = task["key"]
            idx = task["idx"]
            logger_meta = task["logger"]

            # Get transformation data from batch
            if logger_meta is None:
                # -> fallback to the old procedure
                self._log_only_voxel_grid(
                    grid_pred, 2 * idx, f"{log_type}_{key}_prediction"
                )
                self._log_only_voxel_grid(
                    grid_gt, 2 * idx + 1, f"{log_type}_{key}_ground_truth"
                )
            else:
                if task["pair_idx"] is None:
                    # -> all of the images are considered
                    T_cw = logger_meta["T_cw"]
                    K = logger_meta["K"]
                    images = logger_meta["image_paths"]
                    origin = logger_meta["origin"]
                    resolution = logger_meta["pitch"]
                    height, width = (
                        logger_meta["image_size"][0].item(),
                        logger_meta["image_size"][1].item(),
                    )
                else:
                    # we only consider a pair
                    pair_idx = logger_meta["pairs_idxs"][task["pair_idx"]]
                    T_cw = logger_meta["T_cw"][pair_idx]
                    K = logger_meta["K"][pair_idx]
                    images = [logger_meta["image_paths"][idx] for idx in pair_idx]
                    origin = logger_meta["origin"]
                    resolution = logger_meta["resolution"]
                    height, width = (
                        logger_meta["image_size"][0].item(),
                        logger_meta["image_size"][1].item(),
                    )

                if mode == "diff":
                    self._log_voxel_grid_and_images(
                        (grid_gt, grid_pred),
                        2 * idx,
                        f"{log_type}_{key}_prediction",
                        T_cw,
                        K,
                        origin,
                        resolution,
                        images,
                        height,
                        width,
                        type="diff",
                    )

                else:
                    # -> new procedure
                    self._log_voxel_grid_and_images(
                        grid_pred,
                        2 * idx,
                        f"{log_type}_{key}_prediction",
                        T_cw,
                        K,
                        origin,
                        resolution,
                        images,
                        height,
                        width,
                    )
                    self._log_voxel_grid_and_images(
                        grid_gt,
                        2 * idx + 1,
                        f"{log_type}_{key}_ground_truth",
                        T_cw,
                        K,
                        origin,
                        resolution,
                        images,
                        height,
                        width,
                    )
        except Exception as e:
            logger.error(
                f"[OccGridCallbackVisualizer] Failed to process task: {str(e)}"
            )

    def _create_voxel_grid(
        self, grid: Tensor, idx: int
    ) -> Tuple[Optional[wandb.Object3D], Optional[Path]]:
        """Moved from original create_voxel_grid with worker-specific path handling"""
        # grid is a tensor of shape (1, X, Y, Z)

        # create a trimesh object for this grid
        occ = grid
        _1, X, Y, Z = occ.shape

        occ_threshold = 0.5
        encoding = trimesh.voxel.encoding.DenseEncoding(
            (occ > occ_threshold).detach().squeeze(0).bool().cpu().numpy()
        )

        grid = trimesh.voxel.VoxelGrid(encoding)
        occ_re = rearrange(occ, "1 X Y Z -> X Y Z 1")
        # visualize the occ as a opacity
        colors = (
            ((occ_re.float() - occ_threshold) / (1 - occ_threshold))
            .repeat(1, 1, 1, 4)
            .float()
        )
        colors[..., 3] = 1
        if (occ > 0.5).sum() == 0:
            return None, None
        mesh = grid.as_boxes(colors=colors.detach().cpu().numpy())

        tmp_dir = Path(self.wandb.experiment.dir) / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        mesh.export(str(tmp_dir / f"voxel_grid_{idx}.glb"))
        object_3d = wandb.Object3D(str(tmp_dir / f"voxel_grid_{idx}.glb"))
        # delete the tmp file
        file_name = tmp_dir / f"voxel_grid_{idx}.glb"
        return object_3d, file_name

    # def shutdown(self, timeout: float = 5):
    #     """Gracefully stop the worker thread"""
    #     # wait for the queue to be empty
    #     while self.queue.qsize() > 0:
    #         time.sleep(0.1)

    #     self._shutdown_flag.set()
    #     self.thread.join(timeout)
    #     if self.thread.is_alive():
    #         logger.warning("Voxel worker thread did not shutdown gracefully")
