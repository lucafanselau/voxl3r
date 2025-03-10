from multiprocessing import Pool
from pathlib import Path

from loguru import logger
import torch
from tqdm import tqdm
from datasets import scene_processed
from datasets.chunk.mast3r import Dataset as Mast3rDataset, Config as Mast3rConfig
from datasets.chunk.image import Dataset as ImageDataset, Config as ImageConfig


class Config(Mast3rConfig, ImageConfig, scene_processed.Config):
    pass


mast3r_dataset = None


def check_scene(key):
    for path in mast3r_dataset.file_names[key]:
        try:
            temp = torch.load(path, weights_only=False)
        except Exception as e:
            path.unlink()
            print(f"Error loading {path}: {e}")


def main():
    data_config = Config.load_from_files(
        [
            "./config/data/base.yaml",
            "./config/data/mast3r.yaml",
        ]
    )
    # data_config.scenes = ["2e67a32314"]
    data_config.chunk_num_workers = 2
    data_config.split = "train"
    data_config.force_prepare_mast3r = False
    data_config.skip_prepare = True
    data_config.target_chunks = 12000

    base_dataset = scene_processed.Dataset(data_config)
    base_dataset.load_paths()

    # data_config.storage_preprocessing = "preprocessed.bkp"

    image_dataset = ImageDataset(data_config, base_dataset)
    image_dataset.load_paths()
    data_config.mast3r_keys = ["pts3d", "conf", "desc", "desc_conf"]
    global mast3r_dataset
    mast3r_dataset = Mast3rDataset(data_config, base_dataset, image_dataset)
    logger.info("Preparing Mast3r dataset")
    mast3r_dataset.prepare_data()


    scenes = list(mast3r_dataset.file_names.keys())
    with Pool(11) as p:
        with tqdm(
            total=len(scenes), position=0, leave=True
        ) as pbar:
            for _ in p.imap_unordered(check_scene, scenes):
                pbar.update()

    return

    batch_size = 16
    dataloader = torch.utils.data.DataLoader(
        mast3r_dataset, batch_size=batch_size, num_workers=6, persistent_workers=True
    )

    # Helper to create accumulators on first pass
    # Each accumulator is a dict holding:
    # { 'sum': torch.Tensor, 'sum_sq': torch.Tensor, 'count': int }
    def make_accumulator():
        return {"sum": None, "sum_sq": None, "count": 0}

    # Initialize accumulators
    desc1_acc = make_accumulator()
    desc2_acc = make_accumulator()
    desc_conf1_acc = make_accumulator()
    desc_conf2_acc = make_accumulator()
    pts3d1_acc = make_accumulator()
    pts3d2_acc = make_accumulator()
    pts3d_conf1_acc = make_accumulator()
    pts3d_conf2_acc = make_accumulator()

    def accumulate(acc, x):
        """
        Accumulate sum and sum of squares for x.
        x is a flattened 2D tensor: [N, channels] or [N] if no channels.
        """
        x_cpu = x.detach().cpu()
        # If first time, create sum & sum_sq
        if acc["sum"] is None:
            # If x has shape [N, C], we track sums over C
            shape = x_cpu.shape[1:] if x_cpu.dim() > 1 else (1,)
            acc["sum"] = torch.zeros(shape, dtype=torch.float64)
            acc["sum_sq"] = torch.zeros(shape, dtype=torch.float64)

        acc["sum"] += x_cpu.sum(dim=0) if x_cpu.dim() > 1 else x_cpu.sum()
        acc["sum_sq"] += (
            (x_cpu * x_cpu).sum(dim=0) if x_cpu.dim() > 1 else (x_cpu * x_cpu).sum()
        )
        acc["count"] += x_cpu.shape[0]

    for i, batch in tqdm(
        enumerate(dataloader), total=len(mast3r_dataset) // batch_size
    ):
        res1_dict = batch["pairwise_predictions"][0]
        res2_dict = batch["pairwise_predictions"][1]

        # Flatten descriptors: [B, I, W, H, C] -> [(B*I*W*H), C]
        # Flatten confidences: [B, I, W, H] -> [(B*I*W*H)]
        if "desc" in res1_dict:
            desc1_flat = rearrange(res1_dict["desc"], "B I W H C -> (B I W H) C")
            accumulate(desc1_acc, desc1_flat)

        if "desc" in res2_dict:
            desc2_flat = rearrange(res2_dict["desc"], "B I W H C -> (B I W H) C")
            accumulate(desc2_acc, desc2_flat)

        if "desc_conf" in res1_dict:
            desc_conf1_flat = rearrange(res1_dict["desc_conf"], "B I W H -> (B I W H)")
            accumulate(
                desc_conf1_acc, desc_conf1_flat.unsqueeze(-1)
            )  # make it [N,1] for consistency

        if "desc_conf" in res2_dict:
            desc_conf2_flat = rearrange(res2_dict["desc_conf"], "B I W H -> (B I W H)")
            accumulate(desc_conf2_acc, desc_conf2_flat.unsqueeze(-1))

        if "pts3d" in res1_dict:
            pts3d1_flat = rearrange(res1_dict["pts3d"], "B I W H C -> (B I W H) C")
            accumulate(pts3d1_acc, pts3d1_flat)

        if "pts3d" in res2_dict:
            pts3d2_flat = rearrange(res2_dict["pts3d"], "B I W H C -> (B I W H) C")
            accumulate(pts3d2_acc, pts3d2_flat)

        if "pts3d_conf" in res1_dict:
            pts3d_conf1_flat = rearrange(res1_dict["desc_conf"], "B I W H -> (B I W H)")
            accumulate(
                desc_conf1_acc, pts3d_conf1_flat.unsqueeze(-1)
            )  # make it [N,1] for consistency

        if "pts3d_conf" in res2_dict:
            pts3d_conf2_flat = rearrange(res2_dict["desc_conf"], "B I W H -> (B I W H)")
            accumulate(desc_conf2_acc, pts3d_conf2_flat.unsqueeze(-1))

        del batch, res1_dict, res2_dict
        gc.collect()
        torch.cuda.empty_cache()

    # Function to finalize mean/std
    def finalize_stats(acc):
        mean = acc["sum"] / acc["count"]
        var = (acc["sum_sq"] / acc["count"]) - (mean**2)
        std = var.sqrt()
        return mean.float(), std.float()

    # desc1
    desc1_mean, desc1_std = finalize_stats(desc1_acc)
    desc2_mean, desc2_std = finalize_stats(desc2_acc)

    # For confidences, we only used a single channel, so shape is [1].
    desc_conf1_mean, desc_conf1_std = finalize_stats(desc_conf1_acc)
    desc_conf2_mean, desc_conf2_std = finalize_stats(desc_conf2_acc)

    # pts3d
    pts3d1_mean, pts3d1_std = finalize_stats(pts3d1_acc)
    pts3d2_mean, pts3d2_std = finalize_stats(pts3d2_acc)

    # For confidences, we only used a single channel, so shape is [1].
    pts3d_conf1_mean, pts3d_conf1_std = finalize_stats(pts3d_conf1_acc)
    pts3d_conf2_mean, pts3d_conf2_std = finalize_stats(pts3d_conf2_acc)

    # Create a dictionary of final stats
    normalization_stats = {
        "desc1_mean": desc1_mean,
        "desc1_std": desc1_std,
        "desc2_mean": desc2_mean,
        "desc2_std": desc2_std,
        "desc_conf1_mean": desc_conf1_mean[0],  # single value
        "desc_conf1_std": desc_conf1_std[0],
        "desc_conf2_mean": desc_conf2_mean[0],
        "desc_conf2_std": desc_conf2_std[0],
        "pts3d1_mean": pts3d1_mean,
        "pts3d1_std": pts3d1_std,
        "pts3d2_mean": pts3d2_mean,
        "pts3d2_std": pts3d2_std,
        "pts3d_conf1_mean": pts3d_conf1_mean[0],  # single value
        "pts3d_conf1_std": pts3d_conf1_std[0],
        "pts3d_conf2_mean": pts3d_conf2_mean[0],
        "pts3d_conf2_std": pts3d_conf2_std[0],
    }

    # Save everything you need for later normalization
    out_path = (
        Path(data_config.mast3r_data_dir)
        / data_config.storage_preprocessing
        / "normalization_stats_mast3r.pt"
    )
    torch.save(normalization_stats, out_path)


if __name__ == "__main__":
    main()
