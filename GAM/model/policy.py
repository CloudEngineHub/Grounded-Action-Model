"""Flow-matching action policy: ObsEncoder -> ActionTransformer, Euler ODE sampling."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .encoder import ObsEncoder
from .normalizer import Normalizer
from .transformer import ActionTransformer


class GAMPolicy(nn.Module):
    """Args mirror the released checkpoint config (`config` dict inside the .pt file)."""

    def __init__(self, action_dim: int = 14, horizon: int = 16, n_action_steps: int = 16,
                 num_inference_steps: int = 10, n_emb: int = 576, n_layer: int = 12,
                 n_head: int = 16, stream_split=(256, 8, 1), encoder: dict | None = None,
                 image_channel_swap: bool = False, image_wh=(320.0, 240.0),
                 train_image_wh=(1024.0, 768.0), attn_dropout: float = 0.0,
                 lang_film: bool = False, lang_dim: int = 768):
        super().__init__()
        self.lang_film = bool(lang_film)   # language FiLM modulation (see ActionTransformer.lang_film_dim)
        enc = dict(encoder or {})
        enc.setdefault("d", n_emb)
        enc.setdefault("image_wh", image_wh)
        self.encoder = ObsEncoder(**enc)
        assert self.encoder.cond_len == sum(stream_split), (self.encoder.cond_len, stream_split)
        self.model = ActionTransformer(action_dim, horizon, n_emb, self.encoder.cond_len,
                                       stream_split, n_layer=n_layer, n_head=n_head, n_emb=n_emb,
                                       attn_dropout=attn_dropout,
                                       lang_film_dim=(int(lang_dim) if self.lang_film else 0))
        self.normalizer = Normalizer()
        self.action_dim, self.horizon = action_dim, horizon
        self.n_action_steps, self.num_inference_steps = n_action_steps, num_inference_steps
        self.image_channel_swap = bool(image_channel_swap)   # perception hint (see PatchTokenExtractor)
        self.image_wh = tuple(image_wh)
        self.train_image_wh = tuple(train_image_wh)     # 2D boxes must be scaled into this pixel space

    @property
    def device(self):
        return next(self.parameters()).device

    # ---- real-time chunking (Black et al. 2025, "Real-Time Execution of Action Chunking
    # Flow Policies"): inference-time inpainting of the new chunk against the previous chunk's
    # not-yet-executed actions, via pseudoinverse (PiGDM) guidance with a soft prefix mask. ----
    @staticmethod
    def rtc_weights(horizon: int, n_prefix: int, delay: int) -> torch.Tensor:
        """Soft mask W (horizon,) from the paper: 1 on the frozen part (i < delay: these actions
        will be executed while inference runs), exponential decay over the remaining overlap
        (delay <= i < n_prefix), 0 beyond the previous chunk."""
        K = max(0, min(int(n_prefix), horizon))
        d = max(0, min(int(delay), K))
        w = torch.zeros(horizon)
        w[:d] = 1.0
        if K > d:
            i = torch.arange(d, K, dtype=torch.float32)
            c = (K - i) / (K - d + 1)
            w[d:K] = (torch.exp(c) - 1.0) / (math.e - 1.0)
        return w

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, generator=None, prefix: torch.Tensor | None = None,
               weights: torch.Tensor | None = None, guidance_clip: float = 10.0,
               n_frozen: int = 0, lang: torch.Tensor | None = None) -> torch.Tensor:
        """Euler flow sampling from noise (tau=0) to actions (tau=1), all in normalized space.

        prefix  (B, K, Da) normalized target actions for the first K steps (previous chunk's
                remaining actions); weights (horizon,) soft mask from rtc_weights(). When given,
                the velocity is corrected each step by  lambda(tau) * d/dA [ -1/2 |W^1/2 (Y - A1_hat)|^2 ]
                with A1_hat = A_tau + (1 - tau) v  and  lambda = min(beta, (1-tau)/(tau r^2)),
                r^2 = (1-tau)^2 / (tau^2 + (1-tau)^2)  (paper eq.; beta = guidance_clip), with the
                identity-Jacobian approximation of the PiGDM gradient (no backward pass).
        n_frozen: like the reference implementation, the first n_frozen steps (the actions that
                are executed while inference runs) are hard-inpainted each step to the forward-
                noised prefix, on top of the guidance."""
        B = cond.shape[0]
        H, Da = self.horizon, self.action_dim
        x = torch.randn(B, H, Da, device=cond.device, generator=generator)
        N = self.num_inference_steps
        dt = 1.0 / N
        dt_t = torch.tensor([dt], dtype=torch.float32, device=cond.device)
        guided = prefix is not None and weights is not None and float(weights.sum()) > 0
        model = self.model
        if guided:
            K = min(prefix.shape[1], H)
            Y = torch.zeros(B, H, Da, device=cond.device)
            Y[:, :K] = prefix[:, :K].to(cond.device).float()
            Wm = weights.to(cond.device).float().view(1, H, 1)
            nf = max(0, min(int(n_frozen), K))
            x0 = x.clone()
        for i in range(N):
            tau = i / N
            ti = torch.full((B,), tau, device=cond.device)
            if not guided:
                v = model(x, ti, dt_t, cond, lang) if lang is not None else model(x, ti, dt_t, cond)
                x = x.detach().clone() + v * dt
                continue
            r2 = (1 - tau) ** 2 / (tau ** 2 + (1 - tau) ** 2)
            lam = guidance_clip if tau <= 0 else min(guidance_clip, (1 - tau) / (tau * r2))
            v = (model(x, ti, dt_t, cond, lang) if lang is not None else model(x, ti, dt_t, cond)).float()
            v = v + lam * Wm * (Y - (x + (1 - tau) * v))
            x = x.detach() + v * dt
            if nf > 0:                                       # frozen part: forward-noised prefix at tau_next
                tn = (i + 1) / N
                x[:, :nf] = (1 - tn) * x0[:, :nf] + tn * Y[:, :nf]
        return x

    @torch.no_grad()
    def predict_action(self, obs: Dict[str, torch.Tensor], generator=None,
                       prefix: torch.Tensor | None = None, prefix_weights: torch.Tensor | None = None,
                       guidance_clip: float = 10.0, n_frozen: int = 0) -> torch.Tensor:
        """obs: see ObsEncoder.forward (unnormalized). Returns actions (B, n_action_steps, Da).
        prefix: (B, K, Da) UNnormalized committed actions for the first K steps + prefix_weights
        (horizon,) -> real-time-chunking guided sampling (see sample())."""
        nobs = self.normalizer.normalize(obs)
        pn = None
        if prefix is not None and prefix.numel() > 0:
            pn = self.normalizer.normalize({"action": prefix.float()})["action"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond = self.encoder(nobs)
            lang = obs.get("lang") if self.lang_film else None
            x = self.sample(cond, generator, prefix=pn, weights=prefix_weights, guidance_clip=guidance_clip,
                            n_frozen=n_frozen, lang=(lang.to(cond.device) if lang is not None else None))
        act = self.normalizer.unnormalize("action", x[..., :self.action_dim].float())
        return act[:, :self.n_action_steps]

    # ---- serialization -------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda") -> "GAMPolicy":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        meta = cfg.pop("meta", None)
        norm = ck.get("normalizer", {})
        policy = cls(**cfg)
        policy.meta = meta or {}
        for k, v in norm.items():
            policy.normalizer.add(k, torch.as_tensor(v["scale"]), torch.as_tensor(v["offset"]))
        missing, unexpected = policy.load_state_dict(ck["state_dict"], strict=False)
        missing = [m for m in missing if not m.startswith("normalizer.")]
        assert not missing and not unexpected, (missing, unexpected)
        return policy.to(device).eval()

    def save_checkpoint(self, path: str, config: dict):
        norm = {k: {"scale": getattr(self.normalizer, f"{k}_scale").cpu(),
                    "offset": getattr(self.normalizer, f"{k}_offset").cpu()} for k in self.normalizer.keys}
        sd = {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("normalizer.")}   # the EMA may live on the GPU; checkpoints are always CPU tensors
        torch.save({"config": config, "state_dict": sd, "normalizer": norm}, path)
