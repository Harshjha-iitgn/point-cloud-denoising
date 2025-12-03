import torch
import torch.nn as nn
from models.p2pb import P2PB
from models.autoencoder import SemanticAutoencoder
from loguru import logger


class LatentP2PB(P2PB):
    """Robust latent-space P2P-Bridge with frozen Semantic-AE conditioning."""

    def __init__(self, cfg, model=None):
        super().__init__(cfg, model)

        # ---- Load AE ----
        self.ae = SemanticAutoencoder(
            in_dim=getattr(cfg.model, "in_dim", 3),
            feat_dim=getattr(cfg.model, "feat_dim", 512),
            use_dino=getattr(cfg.model, "use_dino", False),
            use_offset_attn=getattr(cfg.model, "use_offset_attn", True),
        )

        ae_ckpt_path = getattr(cfg.model, "ae_ckpt", "")
        try:
            ckpt = torch.load(ae_ckpt_path, map_location="cpu")
            if "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            self.ae.load_state_dict(ckpt, strict=False)
            logger.success(f"Loaded AE weights from {ae_ckpt_path}")
        except Exception as e:
            logger.warning(f"Failed to load AE checkpoint from {ae_ckpt_path}: {e}")

        for p in self.ae.parameters():
            p.requires_grad = False
        self.ae.eval()
        logger.info("Frozen AE initialized for latent conditioning.")

    # ------------------------------------------------------------
    def _to_b3n(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure xyz-only [B,3,N] layout regardless of dataloader shape."""
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,N] or [B,N,C], got {x.shape}")
        B, D1, D2 = x.shape

        if D1 == 3:
            return x
        if D2 == 3:
            return x.transpose(1, 2).contiguous()
        if D1 > 3 and D2 != 3:
            return x[:, :3, :].contiguous()
        if D2 > 3:
            return x.transpose(1, 2)[:, :3, :].contiguous()
        raise RuntimeError(f"Cannot determine xyz layout from {x.shape}")

    # ------------------------------------------------------------
    def forward(self, x0, x1, x_cond=None):
        device = x0.device
        B = x0.shape[0]

        steps = torch.randint(0, self.timesteps, (B,), device=device)
        xt = self.q_sample(steps, x0, x1)
        gt = self.compute_gt(steps, x0, xt)

        if xt.ndim == 3 and xt.shape[1] == 3:
            xt = xt.transpose(1, 2).contiguous()

        # ---- AE latent conditioning ----
        with torch.no_grad():
            xyz = self._to_b3n(x1[:, :3, :])
            latent = self.ae.encode(xyz)  # [B,64,512]

            N = xyz.shape[2]
            if latent.shape[1] != N:
                x_cond = torch.nn.functional.interpolate(
                    latent.transpose(1, 2), size=N, mode="nearest"
                ).transpose(1, 2).contiguous()
            else:
                x_cond = latent.contiguous()

        # ---- Diffusion prediction ----
        try:
            pred_out = self.model(xt, t=steps, x_cond=x_cond)
        except TypeError:
            pred_out = self.model(xt, t=steps)

        pred = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        if pred.shape != gt.shape:
            gt = gt.transpose(1, 2).contiguous()

        loss = self.calculate_loss(pred, gt).mean()
        return loss, latent



