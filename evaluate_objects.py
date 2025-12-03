"""
evaluate_objects.py — final cleaned version for PVDS_PUNet latent AE evaluation
"""

import argparse
import os
from typing import Any, Dict, Generator

import numpy as np
import omegaconf
import pytorch3d.ops
import torch
import trimesh
from loguru import logger
from omegaconf import DictConfig

from models.evaluation import Evaluator, farthest_point_sampling
from models.model_loader import load_diffusion
from models.train_utils import set_seed
from utils.utils import NormalizeUnitSphere, write_array_to_xyz


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/seema/P2PBridge-edit/copy/data/PU_NET/PUNet/pointclouds/test/10000_poisson",
        help="Directory containing noisy inputs (.xyz or .off).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/seema/P2PBridge-edit/copy/eval_output",
        help="Root directory for saving denoised outputs and summaries.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/seema/P2PBridge-edit/copy/data/PU_NET",
        help="Root of dataset for evaluation metrics.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained diffusion checkpoint (e.g., step_30000.pth).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PUNet",
        choices=["PUNet", "PCNet"],
        help="Dataset name for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=3, help="Patch oversampling factor.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for prediction.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate DDPM steps.")
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=5, help="Number of DDPM sampling steps.")
    parser.add_argument("--distribution_type", default="none")
    args = parser.parse_args()

    # --- load model's opt.yaml if present
    cfg_path = os.path.join(os.path.dirname(args.model_path), "opt.yaml")
    if os.path.exists(cfg_path):
        cfg = omegaconf.OmegaConf.load(cfg_path)
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(vars(args)))
    else:
        cfg = omegaconf.OmegaConf.create(vars(args))

    cfg.restart = False
    cfg.local_rank = 0
    return cfg


# ------------------------------------------------------------
# Input iterator (.xyz / .off)
# ------------------------------------------------------------
def input_iter(input_dir: str) -> Generator[Dict[str, Any], None, None]:
    """Yield normalized point clouds from .xyz or .off files."""
    for fn in os.listdir(input_dir):
        fp = os.path.join(input_dir, fn)
        if not os.path.isfile(fp):
            continue

        try:
            if fn.endswith(".xyz"):
                pts = np.loadtxt(fp)
            elif fn.endswith(".off"):
                mesh = trimesh.load_mesh(fp, process=False)
                pts = np.asarray(mesh.vertices)
            else:
                continue
        except Exception as e:
            logger.warning(f"Skipping {fn}: {e}")
            continue

        if pts.ndim != 2 or pts.shape[1] != 3:
            logger.warning(f"Skipping {fn}: invalid shape {pts.shape}")
            continue

        pts = np.nan_to_num(pts)
        pcl_noisy = torch.as_tensor(pts, dtype=torch.float32)
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {"pcl_noisy": pcl_noisy, "name": os.path.splitext(fn)[0], "center": center, "scale": scale}


# ------------------------------------------------------------
# Patch-based denoising
# ------------------------------------------------------------
@torch.no_grad()
def patch_based_denoise(model, pcl_noisy, patch_size, seed_k=3, cfg=None, save_intermediate=False):
    """Patch-based denoising for a single point cloud."""
    assert pcl_noisy.ndim == 2 and pcl_noisy.shape[1] == 3
    N = pcl_noisy.size(0)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # [1, N, 3]

    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # [M, K, 3]

    centers = patches.mean(dim=1, keepdim=True)
    patches = patches - centers
    scale = patches.pow(2).sum(dim=-1).sqrt().max().clamp(min=1e-6)
    patches = patches / scale

    model.eval()
    out = model.sample(
        x_start=patches.transpose(1, 2).contiguous(),
        use_ema=cfg.use_ema,
        steps=cfg.steps,
        log_count=cfg.steps,
        verbose=False,
    )

    patches_denoised = out["x_pred"]
    if patches_denoised.ndim == 3 and patches_denoised.shape[1] == 3:
        patches_denoised = patches_denoised.transpose(1, 2).contiguous()
    elif patches_denoised.shape[-1] != 3:
        patches_denoised = patches_denoised[..., :3]

    patches_steps = out["x_chain"]
    if patches_steps.ndim == 4 and patches_steps.shape[2] == 3:
        patches_steps = patches_steps.transpose(2, 3).contiguous()
    elif patches_steps.shape[-1] != 3:
        patches_steps = patches_steps[..., :3]

    scale = scale.view(1, 1, 1)
    patches_denoised = patches_denoised * scale + centers
    patches_steps = patches_steps * scale.unsqueeze(0) + centers.unsqueeze(1)
    patches_steps = patches_steps.transpose(1, 0)

    pcl_denoised, _ = farthest_point_sampling(patches_denoised.reshape(1, -1, 3), N)
    pcl_denoised = pcl_denoised[0].squeeze()

    pcl_steps_denoised = None
    if save_intermediate:
        T, B, n, d = patches_steps.size()
        patches_steps = patches_steps.reshape(T, B * n, d)
        pcl_steps_denoised, _ = farthest_point_sampling(patches_steps, N)

    return pcl_denoised, pcl_steps_denoised


# ------------------------------------------------------------
# Sampling orchestration
# ------------------------------------------------------------
@torch.no_grad()
def sample(cfg: DictConfig) -> None:
    set_seed(cfg)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    model, _ = load_diffusion(cfg)
    model.eval()

    out_root = os.path.join(cfg.output_root, cfg.dataset)
    os.makedirs(out_root, exist_ok=True)

    # --- Determine input directory
    if os.path.isdir(cfg.data_path):
        input_dir = cfg.data_path
    elif hasattr(cfg, "data") and os.path.isdir(os.path.join(cfg.data.data_dir, "PUNet/meshes/test")):
        input_dir = os.path.join(cfg.data.data_dir, "PUNet/meshes/test")
    else:
        raise FileNotFoundError(f"No valid input directory found. Tried {cfg.data_path}")

    logger.info(f"Evaluating using input_dir: {input_dir}")

    save_title = f"P2PBridge_steps_{cfg.steps}" + ("_ema" if cfg.use_ema else "")
    output_dir = os.path.join(out_root, save_title)
    os.makedirs(output_dir, exist_ok=True)

    # --- Denoising each object
    for data in input_iter(input_dir):
        logger.info(f"Processing {data['name']}")
        pcl_noisy = data["pcl_noisy"].cuda()
        pcl_next, pcl_next_steps = patch_based_denoise(
            model=model,
            pcl_noisy=pcl_noisy,
            patch_size=2048,
            seed_k=cfg.k,
            cfg=cfg,
            save_intermediate=cfg.save_intermediate,
        )

        pcl_denoised = pcl_next.cpu() * data["scale"] + data["center"]

        save_path = os.path.join(output_dir, "pcl", data["name"] + ".xyz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        write_array_to_xyz(save_path, pcl_denoised.numpy())

        if cfg.save_intermediate and pcl_next_steps is not None:
            for step, item in enumerate(pcl_next_steps):
                item = item.cpu() * data["scale"] + data["center"]
                out_path = os.path.join(output_dir, "steps", data["name"], f"{data['name']}_{step}.xyz")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                write_array_to_xyz(out_path, item.numpy())

    # --------------------------------------------------------
    # Evaluation stage (CD, P2F, etc.)
    # --------------------------------------------------------
    gt_candidates = [
        "/home/seema/P2PBridge-edit/copy/data/PU_NET/PUNet/pointclouds/test/50000_poisson",
        "/home/seema/P2PBridge-edit/copy/data/PU_NET/PUNet/pointclouds/test/30000_poisson",
        "/home/seema/P2PBridge-edit/copy/data/PU_NET/PUNet/pointclouds/test/10000_poisson",
    ]
    dataset_root = "/home/seema/P2PBridge-edit/copy/data/PU_NET"

    for path in gt_candidates:
        if os.path.isdir(path):
            logger.info(f"Found ground truth folder: {path}")
            break
    else:
        logger.warning("Could not locate any poisson folder. Using cfg.dataset_root as fallback.")
        dataset_root = cfg.dataset_root

    evaluator = Evaluator(
        output_pcl_dir=os.path.join(output_dir, "pcl"),
        dataset_root=dataset_root,
        dataset=cfg.dataset,
        summary_dir=output_dir,
        experiment_name=save_title,
        device=cfg.gpu,
        res_gts="poisson",
    )
    evaluator.run()


def main():
    opt = parse_args()
    sample(opt)


if __name__ == "__main__":
    main()










