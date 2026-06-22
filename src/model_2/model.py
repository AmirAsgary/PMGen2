"""
MHC-Diff (model_2): PyG denoiser + auxiliary pLDDT & torsion heads + objective.

Trains on the same processed store as model_1 (targets = PMGen/AF2 teacher Cα,
teacher_plddt, and — on the side-chain store — teacher_chi). Side-chain torsions
are trained only when ``teacher_chi`` is present in the batch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import pyg_data as PD                               # noqa: E402
from diffusion import DifferentialDiffusion, center  # noqa: E402
from egnn_pyg import EGNNDenoiser, mlp               # noqa: E402

N_AATYPE, N_SEGMENTS = PD.N_AATYPE, PD.N_SEGMENTS
COORD_SCALE = 15.0


def _groove_terms_pyg(pos_t, pos_p, ptr, pep, tau_mid=8.0, s_buried=1.5,
                      tau_out=8.0, tau_far=18.0, rel=None):
    """Groove-membership geometry on flat PyG coords (Å). Per graph, from per-peptide
    nearest-MHC Cα distance: returns buried[ΣN] (soft 1-in-pocket from the TEACHER,
    gates the peptide coord loss) and a soft-band containment on the PREDICTED coords
    (out-of-pocket peptides pushed into a [tau_out, tau_far] Å shell). Grad flows only
    through pos_p; the label is detached. ``rel`` (per-node, in [0,1]) optionally
    down-weights residues whose predicted coords are unreliable (high-noise steps).
    Geometry is done in fp32 (AMP-safe)."""
    pos_t = pos_t.float()
    pos_p = pos_p.float()
    dev = pos_t.device
    buried = pos_t.new_zeros(pos_t.shape[0])
    cnum = pos_p.new_zeros(())
    cden = pos_p.new_zeros(())
    pep_b = pep.bool()
    for g in range(ptr.numel() - 1):
        a, b = int(ptr[g]), int(ptr[g + 1])
        pe = pep_b[a:b]
        me = ~pe
        if int(pe.sum()) == 0 or int(me.sum()) == 0:
            continue
        idx = torch.arange(a, b, device=dev)[pe]
        with torch.no_grad():
            dt = torch.cdist(pos_t[a:b][pe], pos_t[a:b][me])
            bur = torch.sigmoid((tau_mid - dt.min(-1).values) / s_buried)
        buried[idx] = bur
        nnd_p = torch.cdist(pos_p[a:b][pe], pos_p[a:b][me]).min(-1).values
        # cap the band so a far-off residue can't deliver an unbounded gradient
        band = (F.relu(tau_out - nnd_p) ** 2
                + F.relu(nnd_p - tau_far) ** 2).clamp(max=400.0)
        wout = (1.0 - bur) if rel is None else (1.0 - bur) * rel[idx].float()
        cnum = cnum + (wout * band).sum()
        cden = cden + wout.sum()
    return buried, cnum / cden.clamp_min(1.0)


class MHCDiff(nn.Module):
    def __init__(self, T=200, mhc_scale=0.1, h_dim=128, n_layers=6, k=12,
                 use_cross=True, device="cpu", groove_aware=False,
                 lambda_contain=0.5, groove=(8.0, 1.5, 8.0, 18.0)):
        super().__init__()
        self.diff = DifferentialDiffusion(T=T, mhc_scale=mhc_scale).to(device)
        self.T = T
        # groove-aware loss (no new params -> checkpoints load unchanged): dock
        # in-pocket peptides, expel out-of-pocket ones to a shell, learn pLDDT on all.
        self.groove_aware = bool(groove_aware)
        self.lambda_contain = float(lambda_contain)
        self.groove = tuple(groove)
        self.denoiser = EGNNDenoiser(N_AATYPE, N_SEGMENTS, h_dim=h_dim,
                                     n_layers=n_layers, k=k, use_time=True,
                                     use_cross=use_cross)
        self.aux_net = EGNNDenoiser(N_AATYPE, N_SEGMENTS, h_dim=h_dim,
                                    n_layers=2, k=k, use_time=False, use_cross=False)
        self.plddt_head = mlp([h_dim, h_dim, 1])
        self.torsion_head = mlp([h_dim, h_dim, 8])     # 4 χ × (sin,cos)
        self.to(device)

    def _aux_embed(self, x0, data):
        h0 = self.aux_net.node_features(data, t_frac=None)
        _, h = self.aux_net(center(x0, data.batch), h0, data)
        return h

    def compute_loss(self, data, lambdas=(1.0, 0.25, 0.5, 0.1)
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_pep, l_mhc, l_tor, l_plddt = lambdas
        pep, mhc, batch = data.pep, 1.0 - data.pep, data.batch
        x0 = center(data.pos / COORD_SCALE, batch)
        B = int(batch.max()) + 1
        t = torch.randint(0, self.T, (B,), device=x0.device)

        xt, noise = self.diff.q_sample(x0, t, batch, pep)
        h0 = self.denoiser.node_features(data, t_frac=t.float() / self.T)
        eps_hat, _ = self.denoiser(xt, h0, data)
        se = ((eps_hat - noise) ** 2).sum(-1)                  # [ΣN]
        mhc_coord = (se * mhc).sum() / mhc.sum().clamp_min(1.0)

        contain = x0.new_zeros(())
        if self.groove_aware:
            # gate the peptide ε-loss by groove-membership (only dock in-pocket
            # residues) and add containment on the reconstructed x̂0 (expel the rest).
            acp = self.diff._node_acp(t, batch, pep)              # ᾱ_t per node [ΣN,1]
            # x̂0 = (xt − √(1−ᾱ)·ε̂)/√ᾱ. At high noise √ᾱ→0 sends x̂0 to ~1e4 and NaNs
            # the squared-distance loss, so clamp √ᾱ and bound x̂0 to the physical
            # range (same idea as the sampler clip). `rel`=ᾱ_t down-weights the
            # high-noise steps where x̂0 is meaningless anyway.
            x0_pred = (xt - (1 - acp).clamp_min(0).sqrt() * eps_hat) \
                / acp.sqrt().clamp_min(1e-2)
            x0_pred = x0_pred.clamp(-4.0, 4.0)
            # HARD-gate containment to LOW-noise steps (ᾱ_t>0.5). Where ᾱ_t is small
            # x̂0 is unreliable AND its gradient ∝ 1/√ᾱ_t blows up — that is what kept
            # poisoning the run. Above 0.5 the reconstruction is trustworthy and the
            # gradient factor is <=~1.4x. (soft ᾱ-weighting was not enough.)
            rel = (acp.squeeze(-1) > 0.5).float()
            buried, contain = _groove_terms_pyg(
                data.pos, x0_pred * COORD_SCALE, data.ptr, pep, *self.groove,
                rel=rel)
            pep_w = pep * buried
            pep_coord = (se * pep_w).sum() / pep_w.sum().clamp_min(1.0)
        else:
            pep_coord = (se * pep).sum() / pep.sum().clamp_min(1.0)

        h_clean = self._aux_embed(x0, data)
        plddt_pred = self.plddt_head(h_clean).squeeze(-1).sigmoid() * 100.0
        plddt_se = ((plddt_pred - data.teacher_plddt) / 100.0) ** 2     # [ΣN]
        plddt_loss = plddt_se.mean()
        # peptide-only pLDDT error: the overall mean is dominated by the easy,
        # uniformly-high MHC, so report the peptide separately (the part we care
        # about). RMSE in pLDDT points = 100*sqrt(value).
        pep_plddt = (plddt_se * pep).sum() / pep.sum().clamp_min(1.0)

        if hasattr(data, "teacher_chi"):
            sc = self.torsion_head(h_clean).reshape(-1, 4, 2)
            sc = sc / sc.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # unit circle
            cm = data.teacher_chi_mask                              # [N,4]
            tor = (((sc - data.teacher_chi) ** 2).sum(-1) * cm).sum() \
                / cm.sum().clamp_min(1.0)
        else:
            tor = x0.new_zeros(())

        total = (l_pep * pep_coord + l_mhc * mhc_coord
                 + l_tor * tor + l_plddt * plddt_loss)
        if self.groove_aware:
            total = total + self.lambda_contain * contain
        terms = {"total": float(total), "pep_coord": float(pep_coord),
                 "mhc_coord": float(mhc_coord), "torsion": float(tor),
                 "plddt": float(plddt_loss), "pep_plddt": float(pep_plddt),
                 "contain": float(contain)}
        return total, terms

    def forward(self, data, lambdas=(1.0, 0.25, 0.5, 0.1)):
        # so DistributedDataParallel can hook the training step (DDP syncs grads
        # only through .forward()); plain training calls this too.
        return self.compute_loss(data, lambdas)

    @torch.no_grad()
    def sample(self, data, template_pool: "Optional[PD.MHCTemplatePool]" = None,
               mhc_start_t: Optional[int] = None, sampler: str = "ddim",
               n_steps: int = 25, clip: float = 4.0, mhc_from_truth: bool = False):
        """Generate the complex. Peptide starts from noise; the MHC starts from a
        same-length real template, or (``mhc_from_truth``, used by the structural
        eval) from its own ground-truth coords, else from noise."""
        batch, pep = data.batch, data.pep
        x_init = center(torch.randn_like(data.pos), batch)
        if mhc_from_truth:                               # eval: place peptide given true MHC
            mhc = data.pep < 0.5
            x_init[mhc] = data.pos[mhc] / COORD_SCALE
            x_init = center(x_init, batch)
        elif template_pool is not None:
            ptr = data.ptr                                # graph node boundaries
            for g in range(int(batch.max()) + 1):
                n_mhc = int(data.n_mhc[g]) if torch.is_tensor(data.n_mhc) \
                    else int(data.n_mhc)
                tmpl = template_pool.sample_mhc(n_mhc, device=x_init.device)
                if tmpl is not None:
                    s = int(ptr[g])
                    x_init[s:s + n_mhc] = tmpl / COORD_SCALE
            x_init = center(x_init, batch)

        def eps_fn(xt, t):
            h0 = self.denoiser.node_features(data, t_frac=t.float() / self.T)
            return self.denoiser(xt, h0, data)[0]

        if sampler == "ddim":
            x = self.diff.ddim_sample(eps_fn, x_init, batch, pep, n_steps=n_steps,
                                      clip=clip, start_t=mhc_start_t)
        else:                                            # legacy ancestral DDPM
            x = self.diff.sample(eps_fn, x_init, batch, pep, start_t=mhc_start_t)
        return x * COORD_SCALE


def build_loaders(*a, **k):
    return PD.build_loaders(*a, **k)
