import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from models.autoencoder import SemanticAutoencoder
from dataloaders.punet import PUNetDatasetWrapper as PUNetDataset  

# ---------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------
def chamfer_distance(x, y):
    # x,y: [B,N,3]
    diff_x = torch.cdist(x, y, p=2)  # [B,N,N]
    mins1 = diff_x.min(dim=2)[0]
    mins2 = diff_x.min(dim=1)[0]
    loss = (mins1.mean(dim=1) + mins2.mean(dim=1)).mean()
    return loss


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_autoencoder(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset ----
    train_ds = PUNetDataset(
        root=cfg.data.data_dir,
        split="train",
        npoints=cfg.data.npoints,
        augment=cfg.data.augment,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.bs,
        shuffle=True,
        num_workers=cfg.data.workers,
        drop_last=True,
    )

    # ---- Model ----
    model = SemanticAutoencoder(
        in_dim=3,
        feat_dim=512,
        use_dino=getattr(cfg.model, "use_dino", False), 
        dino_dim=getattr(cfg.model, "dino_dim", 384), 
        sem_dim=getattr(cfg.model, "sem_dim", 128),
        use_cross_attn=getattr(cfg.model, "use_cross_attn", True),
        use_offset_attn=getattr(cfg.model, "use_offset_attn", True),
    ).to(device)

    # ------------------------------------------------------------
    # Learning rate pulled from YAML (default = 1e-4)
    # ------------------------------------------------------------
    lr = getattr(cfg.training, "lr", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    logger.info(f"Autoencoder params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M, LR={lr}")

    # ---- Training ----
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            xyz = batch.get("noisy_points", batch.get("points")).to(device)
            print(xyz.shape)
            gt = batch.get("clean_points", xyz).to(device)
            B, N, _ = xyz.shape

            # Dummy latent-conditioning tensor (3 + 512 input channels)
            latent_cond = torch.zeros(B, N, model.feat_dim, device=device)

            optimizer.zero_grad()
            recon, _ = model(xyz, x_cond=latent_cond)

            loss_cd = chamfer_distance(recon, gt)
            loss_l1 = torch.mean(torch.abs(recon - gt))
            loss = loss_cd + 0.1 * loss_l1

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logger.info(f"Epoch {epoch}: AE loss = {epoch_loss/len(train_loader):.6f}")

        # ---- Save checkpoint ----
        os.makedirs(cfg.training.ckpt_dir, exist_ok=True)
        if (epoch % cfg.training.save_interval == 0) or (epoch == cfg.training.epochs):
            path = os.path.join(cfg.training.ckpt_dir, f"ae_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)
            logger.success(f"Saved checkpoint: {path}")

    logger.info("Pretraining complete.")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    from types import SimpleNamespace
    def to_ns(d): 
        return SimpleNamespace(**{k: to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    cfg = to_ns(cfg)

    if not hasattr(cfg.training, "epochs"): cfg.training.epochs = 150
    if not hasattr(cfg.training, "save_interval"): cfg.training.save_interval = 10
    if not hasattr(cfg.training, "ckpt_dir"): cfg.training.ckpt_dir = "checkpoints_ae"

    train_autoencoder(cfg)


