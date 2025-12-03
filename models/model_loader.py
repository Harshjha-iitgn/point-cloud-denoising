from typing import Dict
import os
import torch
from ema_pytorch import EMA
from loguru import logger
from torch import optim
from torch.nn.parallel import DataParallel, DistributedDataParallel

from models.p2pb import P2PB
from models.unet_pvc import PVCNN2Unet
from models.autoencoder import SemanticAutoencoder
from models.p2pb_latent import LatentP2PB


# ------------------------------------------------------------
#  Optimizer + Scheduler loader
# ------------------------------------------------------------
def load_optim_sched(cfg: Dict, model: torch.nn.Module, model_ckpt: str = None) -> tuple:
    """Prepare optimizer and scheduler; restore if checkpoint present."""
    if cfg.training.optimizer.type == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )
    elif cfg.training.optimizer.type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            betas=(cfg.training.optimizer.beta1, cfg.training.optimizer.beta2),
        )
    else:
        raise NotImplementedError(cfg.training.optimizer.type)

    # LR scheduler
    if cfg.training.scheduler.type == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.training.scheduler.lr_gamma)
    elif cfg.training.scheduler.type == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10_000, gamma=0.9)
    else:
        lr_scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    if model_ckpt is not None and not cfg.restart:
        try:
            optimizer.load_state_dict(model_ckpt["optimizer_state"])
        except Exception as e:
            logger.warning(e)

    logger.info("Optimizer and scheduler prepared")
    return optimizer, lr_scheduler


# ------------------------------------------------------------
#  Model loader (PVCNN2Unet / SemanticAutoencoder / LatentP2PB)
# ------------------------------------------------------------
def load_model(cfg: Dict) -> torch.nn.Module:
    """Select and initialize model backbone based on cfg.model.model_type"""
    model_type = str(getattr(cfg.model, "model_type", getattr(cfg.model, "type", "pvcnn"))).lower()

    if model_type in ["semantic_ae", "autoencoder", "latentae"]:
        model = SemanticAutoencoder(
            in_dim=3,
            feat_dim=getattr(cfg.model, "feat_dim", 512),
            use_dino=getattr(cfg.model, "use_dino", False),
            dino_dim=getattr(cfg.model, "dino_dim", 384),
            sem_dim=getattr(cfg.model, "sem_dim", 128),
            use_cross_attn=getattr(cfg.model, "use_cross_attn", False),
            use_offset_attn=getattr(cfg.model, "use_offset_attn", True),
        )
        logger.info("Loaded Semantic-Attention Autoencoder backbone.")

        # ------------------------------------------------------------
        #  Load retrained AE checkpoint if available
        # ------------------------------------------------------------
        ae_ckpt_path = getattr(cfg.model, "ae_ckpt", "")
        logger.info(f"Using AE checkpoint path: {ae_ckpt_path}")
        logger.info(f"Using AE checkpoint path: {ae_ckpt_path}")
        if ae_ckpt_path and os.path.exists(ae_ckpt_path):
            try:
                ckpt = torch.load(ae_ckpt_path, map_location="cpu")
                if "model_state_dict" in ckpt:
                    ckpt = ckpt["model_state_dict"]
                model.load_state_dict(ckpt, strict=False)
                logger.success(f"Loaded pretrained AE weights from {ae_ckpt_path}")
            except Exception as e:
                logger.warning(f"Could not load AE checkpoint from {ae_ckpt_path}: {e}")
        else:
            logger.warning(f"AE checkpoint not found at {ae_ckpt_path}")

    elif model_type in ["latentp2pb", "latent_p2pb"]:
        # ✅ FIX: Always give LatentP2PB a valid backbone (PVCNN2Unet)
        base_backbone = PVCNN2Unet(cfg)
        model = LatentP2PB(cfg, model=base_backbone)
        logger.info("Loaded LatentP2PB diffusion bridge with PVCNN2Unet backbone.")

    else:
        model = PVCNN2Unet(cfg)
        logger.info("Loaded PVCNN2Unet backbone.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Generated model with following number of params (M): {num_params:.2f}")
    return model


# ------------------------------------------------------------
#  Diffusion model loader and wrapper setup
# ------------------------------------------------------------
def load_diffusion(cfg: Dict) -> tuple:
    """Setup full diffusion model (bridge + backbone) and load checkpoint."""
    backbone = load_model(cfg).to(cfg.local_rank)

    # If using SemanticAutoencoder, use the latent bridge
    if getattr(cfg.model, "model_type", "") == "semantic_ae":
        model = LatentP2PB(cfg=cfg, model=backbone)
    else:
        model = P2PB(cfg=cfg, model=backbone)

    gpu = cfg.local_rank
    model = model.cuda()

    # Distributed / DataParallel handling
    if cfg.distribution_type == "multi":
        def ddp_transform(m):
            return DistributedDataParallel(m, device_ids=[gpu], output_device=gpu)
        model.multi_gpu_wrapper(ddp_transform)

    elif cfg.distribution_type == "single":
        def dp_transform(m):
            return DataParallel(m)
        model.multi_gpu_wrapper(dp_transform)

    # ------------------------------------------------------------
    #  Load model checkpoint if provided
    # ------------------------------------------------------------
    cfg.start_step = 0
    ckpt = None
    if cfg.model_path != "":
        ckpt = torch.load(cfg.model_path, map_location=torch.device("cpu"))

        if not cfg.restart:
            cfg.start_step = ckpt.get("step", 0) + 1
            try:
                model_state = ckpt["model_state"]
                if cfg.distribution_type in ["multi", "single"]:
                    model_dict = extract_from_state_dict(model_state, "model.")
                    ema_dict = extract_from_state_dict(model_state, "ema.")
                else:
                    model_dict = extract_from_state_dict(model_state, "model.module.")
                    ema_dict = extract_from_state_dict(model_state, "ema.")

                model.model.load_state_dict(model_dict, strict=False)
                if cfg.use_ema and ema_dict != {}:
                    model.ema.load_state_dict(ema_dict)
                    logger.success("Loaded EMA from checkpoint!")
                logger.success("Loaded Model from checkpoint!")

            except RuntimeError as e:
                logger.warning("Could not load model state dict. Trying to load without strict flag.")
                logger.warning(e)
                model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            logger.info("Restarting training: new optimizer, new EMA.")
            model_state = ckpt["model_state"]
            model_dict = extract_from_state_dict(model_state, "model.")
            try:
                model.model.load_state_dict(model_dict, strict=False)
                if cfg.use_ema:
                    model.ema = EMA(model.model, beta=0.999)
            except RuntimeError:
                logger.warning("Adaptive weight loading triggered.")
                load_matched_weights(model.model, model_dict)
                if cfg.use_ema:
                    model.ema = EMA(model.model, beta=0.999)
        logger.info(f"Loaded model from {cfg.model_path}")

    torch.cuda.empty_cache()
    return model, ckpt


# ------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------
def extract_from_state_dict(state_dict: Dict, pattern: str) -> Dict:
    """Extract subkeys from state_dict whose name starts with pattern."""
    return {k.replace(pattern, ""): v for k, v in state_dict.items() if k.startswith(pattern)}


def load_matched_weights(model: torch.nn.Module, state_dict_to_load: Dict):
    """Load only matching weights (safe load)."""
    own_state = model.state_dict()
    for name, param in state_dict_to_load.items():
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if name in own_state and own_state[name].shape == param.shape:
            own_state[name].copy_(param)
        elif "." in name:
            sub_names = name.split(".")
            sub_module = model
            for sn in sub_names[:-1]:
                if hasattr(sub_module, sn):
                    sub_module = getattr(sub_module, sn)
                else:
                    break
            else:
                sub_param_name = sub_names[-1]
                if hasattr(sub_module, sub_param_name):
                    sub_param = getattr(sub_module, sub_param_name)
                    if sub_param.shape == param.shape:
                        sub_param.data.copy_(param)
        else:
            logger.debug(f"Skipped unmatched parameter: {name}")




