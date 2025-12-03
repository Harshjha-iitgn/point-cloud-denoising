from functools import partial
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention
from .pvcnn import (
    LinearAttention,
    Pnet2Stage,
    PVCData,
    SharedMLP,
    Swish,
    create_fp_components,
    create_mlp_components,
    create_pvc_layer_params,
    create_sa_components,
)

# adapted from https://github.com/alexzhou907/PVD
class PVCNN2Unet(nn.Module):
    def __init__(self, cfg: Dict, return_layers: bool = False):
        super().__init__()

        model_cfg = cfg.model
        pvd_cfg = model_cfg.PVD

        self.return_layers = return_layers
        self.input_dim = train_utils.default(model_cfg.in_dim, 3)

        # ---- FIX: always define this field ----
        self.extra_feature_channels = 0
        if "extra_feature_channels" in pvd_cfg:
            self.extra_feature_channels = pvd_cfg.extra_feature_channels
        elif "extra_feature_channels" in model_cfg:
            self.extra_feature_channels = model_cfg.extra_feature_channels
        # ---------------------------------------

        self.embed_dim = train_utils.default(model_cfg.time_embed_dim, 64)
        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)
        attn_type = train_utils.default(pvd_cfg.attention_type, "linear")

        self.embedf = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # global embedding
        if pvd_cfg.get("use_global_embedding", False):
            self.cond_emb_dim = pvd_cfg.global_embedding_dim
            c = self.cond_emb_dim
            self.global_pnet = Pnet2Stage([self.input_dim, c // 8, c // 4], [c // 2, c])
        else:
            self.global_pnet = None
            self.cond_emb_dim = 0

        self.f_embed_dim = pvd_cfg.get("feat_embed_dim", self.extra_feature_channels)

        self.embed_feats = None
        if self.f_embed_dim != self.extra_feature_channels:
            in_dim = self.extra_feature_channels if self.extra_feature_channels > 0 else self.input_dim
            self.embed_feats = nn.Sequential(
                nn.Conv1d(in_dim, self.f_embed_dim, kernel_size=1, bias=True),
                nn.GroupNorm(8, self.f_embed_dim),
                Swish(),
                nn.Conv1d(self.f_embed_dim, self.f_embed_dim, kernel_size=1, bias=True),
            )

        sa_blocks, fp_blocks = create_pvc_layer_params(
            npoints=cfg.data.npoints,
            channels=cfg.model.PVD.channels,
            n_sa_blocks=cfg.model.PVD.n_sa_blocks,
            n_fp_blocks=cfg.model.PVD.n_fp_blocks,
            radius=cfg.model.PVD.radius,
            voxel_resolutions=cfg.model.PVD.voxel_resolutions,
            centers=pvd_cfg.get("centers", None),
        )

        # prepare attention
        attn_type_lower = attn_type.lower()
        if attn_type_lower == "linear":
            attention_fn = partial(LinearAttention, heads=cfg.model.PVD.attention_heads)
        elif attn_type_lower == "flash":
            attention_fn = partial(Attention, norm=False, flash=True, heads=cfg.model.PVD.attention_heads)
        else:
            attention_fn = None

        # create set abstraction layers
        sa_layers, sa_in_channels, channels_sa_features, *_ = create_sa_components(
            input_dim=self.input_dim,
            sa_blocks=sa_blocks,
            extra_feature_channels=self.f_embed_dim,
            with_se=pvd_cfg.get("use_se", True),
            embed_dim=self.embed_dim,
            attention_fn=attention_fn,
            attention_layers=cfg.model.PVD.attentions,
            dropout=dropout,
            gn_groups=8,
            cond_dim=self.cond_emb_dim,
        )

        self.sa_layers = nn.ModuleList(sa_layers)

        if attention_fn is not None:
            self.global_att = attention_fn(dim=channels_sa_features)
        else:
            self.global_att = None

        # create feature propagation layers
        sa_in_channels[0] = self.f_embed_dim + self.input_dim
        fp_layers, channels_fp_features = create_fp_components(
            fp_blocks=fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=pvd_cfg.get("use_se", True),
            embed_dim=self.embed_dim,
            attention_layers=cfg.model.PVD.attentions,
            attention_fn=attention_fn,
            dropout=dropout,
            gn_groups=8,
            cond_dim=self.cond_emb_dim,
        )

        self.fp_layers = nn.ModuleList(fp_layers)

        out_mlp = cfg.model.PVD.get("out_mlp", 128)
        layers, *_ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[out_mlp, dropout, out_dim],
            classifier=True,
            dim=2,
        )
        self.classifier = nn.ModuleList(layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb

    def forward(self, x, t, x_cond=None):
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)

        B, C, N = x.shape
        device = x.device
        # ---- FIX: avoid crash when extra_feature_channels == 0 ----
        if not hasattr(self, "extra_feature_channels"):
            self.extra_feature_channels = 0
        # -----------------------------------------------------------
        expected = self.input_dim + self.extra_feature_channels
        assert C == expected, f"input dim: {C}, expected: {expected}"

        coords = x[:, : self.input_dim, :].contiguous()
        features = x[:, self.input_dim :, :].contiguous()

        if self.embed_feats is not None:
            features = self.embed_feats(coords if self.extra_feature_channels == 0 else features)

        data = PVCData(coords=coords, features=coords)

        if self.global_pnet is not None:
            global_feature = self.global_pnet(data)
            data.cond = global_feature

        features = torch.cat([coords, features], dim=1)
        coords_list, in_features_list, out_features_list = [], [], []
        in_features_list.append(features)

        time_emb = None
        if t is not None:
            if t.ndim == 0 or len(t.shape) != 1:
                t = t.view(-1)
            time_emb = self.embedf(self.get_timestep_embedding(t, device))[:, :, None].expand(-1, -1, N)

        data.features = features
        data.time_emb = time_emb

        for sa_blocks in self.sa_layers:
            in_features_list.append(data.features)
            coords_list.append(data.coords)
            if data.time_emb is not None:
                data.features = torch.cat([data.features, data.time_emb], dim=1)
            data = sa_blocks(data)

        in_features_list.pop(1)

        if self.global_att is not None:
            feats = data.features
            if isinstance(self.global_att, LinearAttention):
                feats = self.global_att(feats)
            elif isinstance(self.global_att, Attention):
                feats = rearrange(feats, "b n c -> b c n")
                feats = self.global_att(feats)
                feats = rearrange(feats, "b c n -> b n c")
            data.features = feats

        out_features_list.append(data.features)

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            data_fp = PVCData(
                features=in_features_list[-1 - fp_idx],
                coords=coords_list[-1 - fp_idx],
                lower_coords=data.coords,
                lower_features=(
                    torch.cat([data.features, data.time_emb], dim=1)
                    if data.time_emb is not None else data.features
                ),
                time_emb=data.time_emb,
                cond=getattr(data, "cond", None),
            )
            data = fp_blocks(data_fp)
            out_features_list.append(data.features)

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                data.features = l(data).features
            else:
                data.features = l(data.features)

        return data.features

