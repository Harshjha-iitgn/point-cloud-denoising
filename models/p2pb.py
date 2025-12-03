# models/p2pb.py

from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ema_pytorch import EMA
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from models.train_utils import DiffusionModel
from .loss import get_loss


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def space_indices(num_steps: int, count: int):
    """Evenly spaced indices from [0, num_steps-1] of length count."""
    assert count <= num_steps
    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride
    return taken_steps


def extract(a: Tensor, t: Tensor, x_shape: Tuple[int, ...]):
    """Gather a[t] and reshape for broadcast to x_shape (batch-first)."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def unsqueeze_xdim(z: Tensor, xdim: Tuple[int, ...]) -> Tensor:
    """Expand scalar/batch tensor z to match xdim for broadcasting."""
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def compute_gaussian_product_coef(sigma1: float, sigma2: float):
    """Closed-form product of two zero-mean Gaussians N(0,s1^2) and N(0,s2^2)."""
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def make_beta_schedule(n_timestep: int = 1000, linear_start: float = 1e-4, linear_end: float = 2e-2):
    """Cosine-like linear-in-sqrt schedule used in P2P-Bridge codebase."""
    scale = 1000 / n_timestep
    linear_start *= scale
    linear_end *= scale
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.numpy()


# ---------------------------------------------------------------------
# Shape alignment utilities
# ---------------------------------------------------------------------

def align_to_bnc(x: Tensor) -> Tensor:
    """Ensure tensor is [B, N, 3]."""
    if x.ndim == 3 and x.shape[1] == 3 and x.shape[2] != 3:
        return x.transpose(1, 2).contiguous()
    return x


def align_pair(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Ensure both tensors are [B, N, 3] and same shape."""
    a = align_to_bnc(a)
    b = align_to_bnc(b)
    if a.shape != b.shape:
        min_n = min(a.shape[1], b.shape[1])
        a = a[:, :min_n, :]
        b = b[:, :min_n, :]
    return a, b


# ---------------------------------------------------------------------
# P2PB Diffusion Wrapper
# ---------------------------------------------------------------------

class P2PB(DiffusionModel):
    """
    Diffusion bridge for PVCNN2UNet / Semantic Autoencoder backbones.
    Fully shape-robust: auto-aligns all 3D tensors to [B, N, 3].
    """

    def __init__(self, cfg: Dict, model: torch.nn.Module):
        super().__init__()

        # ---- config ----
        device = cfg.gpu if hasattr(cfg, "gpu") and cfg.gpu is not None else torch.device("cuda")
        self.device = device
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.ot_ode = cfg.diffusion.ot_ode
        self.cfg = cfg
        self.cond_x1 = cfg.diffusion.get("cond_x1", False)
        self.add_x1_noise = cfg.diffusion.get("add_x1_noise", False)
        self.objective = cfg.diffusion.get("objective", "pred_noise")
        self.weight_loss = cfg.diffusion.get("weight_loss", False)
        self.symmetric = cfg.diffusion.get("symmetric", True)
        self.loss_multiplier = cfg.diffusion.get("loss_multiplier", 1.0)
        snr_clip = cfg.diffusion.get("snr_clip", False)

        # ---- model + EMA ----
        self.model = model.to(device)
        self.ema = EMA(self.model, beta=0.999) if getattr(cfg.model, "ema", False) else None

        # ---- schedule ----
        betas = make_beta_schedule(
            n_timestep=cfg.diffusion.timesteps,
            linear_start=cfg.diffusion.beta_start,
            linear_end=cfg.diffusion.beta_end,
        )
        if self.symmetric:
            betas = np.concatenate(
                [betas[: cfg.diffusion.timesteps // 2], np.flip(betas[: cfg.diffusion.timesteps // 2])]
            )

        # noise levels
        self.noise_levels = (
            torch.linspace(cfg.diffusion.t0, cfg.diffusion.T, cfg.diffusion.timesteps, dtype=torch.float32).to(device)
            * cfg.diffusion.timesteps
        )

        # forward/backward stds
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensors
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)
        self.calculate_loss = get_loss(cfg.diffusion.get("loss_type", "mse"))

        # SNR weighting (optional)
        alphas_cumprod = np.cumprod(1 - betas)
        snr = torch.from_numpy(alphas_cumprod / (1 - alphas_cumprod)).to(device)
        maybe_clipped_snr = snr.clone()
        if snr_clip:
            maybe_clipped_snr.clamp_(max=5.0)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)
        if self.objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif self.objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_std_fwd(self, step: Tensor, xdim: Tuple[int, ...] = None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def compute_pred_x0_from_eps(self, step: Tensor, xt: Tensor, net_out: Tensor, clip_denoise: bool = False):
        """Compute predicted x0 ensuring both tensors are [B,N,3]."""
        xt, net_out = align_pair(xt, net_out)
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
        if std_fwd.ndim != net_out.ndim:
            std_fwd = std_fwd.squeeze(-1)

        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-3.0, 3.0)
        return pred_x0

    def compute_gt(self, step: Tensor, x0: Tensor, xt: Tensor) -> Tensor:
        x0, xt = align_pair(x0, xt)
        if self.objective == "pred_noise":
            std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
            gt = (xt - x0) / std_fwd
            return gt.detach()
        elif self.objective == "pred_x0":
            return x0.detach()

    def q_sample(self, step: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        x0, x1 = align_pair(x0, x1)
        _, *xdim = x0.shape
        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)
        xt = mu_x0 * x0 + mu_x1 * x1
        if not self.ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev: int, n: int, x_n: Tensor, x0: Tensor) -> Tensor:
        """Posterior p(x_{t-1} | x_t, x_0)."""
        assert nprev < n
        x_n, x0 = align_pair(x_n, x0)

        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()
        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not self.ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
        return xt_prev

    # ------------------------------------------------------------------
    # Sampling core
    # ------------------------------------------------------------------
    def sample_ddpm(
        self,
        steps: List[int],
        pred_x0_fn: callable,
        x1: Tensor,
        x_cond: Tensor = None,
        log_steps: List[int] = None,
        verbose: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        x1 = align_to_bnc(x1)
        if self.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)

        xt = x1.detach().to(self.device)
        xs, pred_x0s = [], []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]
        iterator = zip(steps[1:], steps[:-1])
        if verbose:
            iterator = tqdm(iterator, desc="DDPM sampling", total=len(steps) - 1)

        for prev_step, step in iterator:
            pred_x0 = pred_x0_fn(xt, step, x1=x1, x_cond=x_cond)
            xt = self.p_posterior(prev_step, step, xt, pred_x0)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach())
                xs.append(xt.detach())

        stack = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack(xs), stack(pred_x0s)

    @torch.no_grad()
    def ddpm_sampling(
        self,
        x1: Tensor,
        x_cond: Tensor = None,
        clip_denoise: bool = False,
        sampling_steps: int = None,
        log_count: int = 10,
        verbose: bool = True,
        use_ema: bool = False,
    ) -> Tuple[Tensor, Tensor]:

        x1 = align_to_bnc(x1)
        sampling_steps = sampling_steps or self.timesteps - 1
        assert 0 < sampling_steps < self.timesteps == len(self.betas)

        steps = space_indices(self.timesteps, sampling_steps + 1)
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0

        if verbose:
            logger.info(f"[DDPM Sampling] T={self.timesteps}, sampling_steps={sampling_steps}!")

        self.model.eval()

        def pred_x0_fn(xt, step, x1, x_cond=None):
            xt = align_to_bnc(xt)
            x1 = align_to_bnc(x1)

            step = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)
            noise_levels = self.noise_levels[step].detach()

            if self.cond_x1:
                x_cond = torch.cat([x1, x_cond], dim=1)

            out = self.ema(xt, noise_levels, x_cond=x_cond) if (use_ema and self.ema is not None) else self.model(
                xt, noise_levels, x_cond=x_cond
            )
            if isinstance(out, tuple):
                out = out[0]
            out = align_to_bnc(out)
            return self.compute_pred_x0_from_eps(step, xt, out, clip_denoise=clip_denoise)

        xs, pred_x0 = self.sample_ddpm(steps, pred_x0_fn, x1, x_cond=x_cond, log_steps=log_steps, verbose=verbose)

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        self.model.train()
        return xs, pred_x0

    @torch.no_grad()
    def sample(
        self,
        x_cond: Optional[Tensor] = None,
        x_start: Optional[Tensor] = None,
        clip: bool = False,
        use_ema: bool = False,
        verbose: bool = True,
        log_count: int = 10,
        steps: int = None,
    ) -> Dict:
        if self.cfg.diffusion.sampling_strategy == "DDPM":
            xs, x0s = self.ddpm_sampling(
                x1=x_start,
                x_cond=x_cond,
                clip_denoise=clip,
                sampling_steps=self.cfg.diffusion.sampling_timesteps if steps is None else steps,
                verbose=verbose,
                use_ema=use_ema,
                log_count=log_count,
            )
            return {"x_chain": xs, "x_pred": xs[:, 0, ...], "x_start": x_start}
        return {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        pred, gt = align_pair(pred, gt)
        pred = pred.to(self.device)
        gt = gt.to(self.device)
        loss = self.calculate_loss(pred, gt)
        return loss.mean() * self.loss_multiplier

    def forward(self, x0: Tensor, x1: Tensor, x_cond: Optional[Tensor] = None) -> Tensor:
        x0, x1 = align_pair(x0, x1)
        steps = torch.randint(0, self.timesteps, (x0.shape[0],), device=self.device)

        if self.add_x1_noise:
            x1 = x1 + torch.randn_like(x1)

        xt = self.q_sample(steps, x0, x1)
        gt = self.compute_gt(steps, x0, xt)

        if self.cond_x1:
            x_cond = torch.cat([x1, x_cond], dim=1) if x_cond is not None else x1

        noise_levels = self.noise_levels[steps].detach()
        pred = self.model(xt, noise_levels, x_cond=x_cond)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = align_to_bnc(pred)

        loss = self.calculate_loss(pred, gt)
        if self.weight_loss:
            loss = loss * extract(self.loss_weight, steps, loss.shape)
        return loss.mean() * self.loss_multiplier





