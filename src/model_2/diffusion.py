"""
Differential, zero-CoM coordinate diffusion for MHC-Diff (model_2, PyG form).

Operates on **flattened** node coordinates ``pos [ΣN, 3]`` with a PyG ``batch``
vector (one graph per example, variable N, no padding). Differential schedules:
peptide = full-variance cosine; MHC = low-variance (induced fit). Everything lives
in the per-graph zero centre-of-mass subspace (centred with ``scatter_mean``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch_scatter import scatter_mean


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    acp = f / f[0]
    betas = 1 - acp[1:] / acp[:-1]
    return betas.clamp(1e-6, 0.999).float()


def center(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Subtract each graph's centre of mass. pos[ΣN,3], batch[ΣN]."""
    return pos - scatter_mean(pos, batch, dim=0)[batch]


class DifferentialDiffusion:
    def __init__(self, T: int = 200, mhc_scale: float = 0.1):
        self.T = T
        betas = cosine_betas(T)
        self.betas = {"pep": betas, "mhc": betas * mhc_scale}
        self.acp = {k: torch.cumprod(1.0 - b, 0) for k, b in self.betas.items()}
        self.device = "cpu"

    def to(self, device):
        self.betas = {k: v.to(device) for k, v in self.betas.items()}
        self.acp = {k: v.to(device) for k, v in self.acp.items()}
        self.device = device
        return self

    def _node_acp(self, t, batch, pep):
        """Per-node α̅ from per-graph t[B], node batch[ΣN], node pep[ΣN]."""
        tn = t[batch]                                            # [ΣN]
        acp = torch.where(pep.bool(), self.acp["pep"][tn], self.acp["mhc"][tn])
        return acp[:, None]                                      # [ΣN,1]

    def q_sample(self, x0, t, batch, pep, noise=None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = center(x0, batch)
        if noise is None:
            noise = torch.randn_like(x0)
        noise = center(noise, batch)
        acp = self._node_acp(t, batch, pep)
        xt = acp.sqrt() * x0 + (1 - acp).sqrt() * noise
        return center(xt, batch), noise

    @torch.no_grad()
    def p_sample_step(self, eps_fn, xt, t, batch, pep):
        eps = eps_fn(xt, t)
        tn = t[batch]
        acp = self._node_acp(t, batch, pep)
        beta = torch.where(pep.bool(), self.betas["pep"][tn],
                           self.betas["mhc"][tn])[:, None]
        alpha = 1.0 - beta
        mean = (xt - beta / (1 - acp).clamp_min(1e-8).sqrt() * eps) / alpha.sqrt()
        nz = (t[batch] > 0).float()[:, None]
        x_prev = mean + nz * beta.sqrt() * center(torch.randn_like(xt), batch)
        return center(x_prev, batch)

    @torch.no_grad()
    def sample(self, eps_fn, x_init, batch, pep, start_t: Optional[int] = None):
        """Ancestral DDPM reverse process (stochastic, all T steps). Kept for
        reference; ``ddim_sample`` is the default — faster and far more stable."""
        x = center(x_init, batch)
        t0 = (self.T - 1) if start_t is None else min(start_t, self.T - 1)
        for ti in range(t0, -1, -1):
            t = torch.full((int(batch.max()) + 1,), ti, device=x.device,
                           dtype=torch.long)
            x = self.p_sample_step(eps_fn, x, t, batch, pep)
        return x

    @torch.no_grad()
    def ddim_sample(self, eps_fn, x_init, batch, pep, n_steps: int = 25,
                    clip: float = 4.0, start_t: Optional[int] = None):
        """Deterministic DDIM (eta=0) over a strided ``n_steps`` schedule, with
        per-step **x0-clipping** (dynamic-thresholding style) — this is what tames
        the divergence the ancestral sampler showed. ``clip`` is in normalized
        (÷COORD_SCALE, CoM-centred) units; ~4 ≈ ±60 Å. ``n_steps`` ~10-25 is enough
        and ~T/n_steps faster than full DDPM."""
        T = self.T
        t0 = (T - 1) if start_t is None else min(start_t, T - 1)
        ts = torch.linspace(t0, 0, n_steps + 1).round().long().to(x_init.device)
        B = int(batch.max()) + 1
        x = center(x_init, batch)
        for i in range(n_steps):
            tb = torch.full((B,), int(ts[i]), device=x.device, dtype=torch.long)
            tp = torch.full((B,), int(ts[i + 1]), device=x.device, dtype=torch.long)
            eps = eps_fn(x, tb)
            acp_t = self._node_acp(tb, batch, pep)
            acp_p = self._node_acp(tp, batch, pep)
            x0 = (x - (1 - acp_t).clamp_min(0).sqrt() * eps) / acp_t.sqrt().clamp_min(1e-4)
            if clip:
                x0 = x0.clamp(-clip, clip)
            x0 = center(x0, batch)
            x = center(acp_p.sqrt() * x0 + (1 - acp_p).clamp_min(0).sqrt() * eps, batch)
        return x
