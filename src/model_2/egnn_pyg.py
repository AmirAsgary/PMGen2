"""
SE(3)-equivariant EGNN in PyTorch Geometric (model_2 backbone).

Graphs are built with ``torch_cluster.radius_graph`` (type-dependent cutoffs) for
the messages, and a bounded ``knn_graph`` for the triplet/chirality term (keeps
the O(N³) cross-product to O(N·k²)). Message passing is scatter-based (the PyG
idiom). The cross-product coordinate update breaks E(3)→SE(3) (chirality-aware).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn, knn_graph, radius_graph
from torch_scatter import scatter


def sinusoidal(v: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=v.device) / max(half - 1, 1))
    a = v[..., None].float() * freqs
    emb = torch.cat([a.sin(), a.cos()], -1)
    return F.pad(emb, (0, dim - emb.shape[-1])) if emb.shape[-1] < dim else emb


def mlp(sizes, act=nn.SiLU, zero_last=False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    net = nn.Sequential(*layers)
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def _intra_group_pairs(items, groups, n_groups):
    """All directed pairs (a,b), a≠b, of items sharing a group. Vectorized."""
    m = items.shape[0]
    if m == 0:
        return items.new_zeros(2, 0)
    order = torch.argsort(groups)
    g_sorted = groups[order]
    counts = torch.bincount(g_sorted, minlength=n_groups)
    starts = torch.cumsum(counts, 0) - counts
    rep = counts[g_sorted]
    gs = starts[g_sorted]
    total = int(rep.sum())
    if total == 0:
        return items.new_zeros(2, 0)
    dev = items.device
    a_s = torch.repeat_interleave(torch.arange(m, device=dev), rep)
    cum = torch.cumsum(rep, 0) - rep
    local = torch.arange(total, device=dev) - torch.repeat_interleave(cum, rep)
    b_s = torch.repeat_interleave(gs, rep) + local
    keep = a_s != b_s
    return torch.stack([items[order[a_s[keep]]], items[order[b_s[keep]]]], 0)


def build_message_edges(pos, batch, pep, residue_index, cut_pp=8.0, cut_pm=14.0,
                        max_nb=48, pep_mhc_k=6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Message graph (directed, deduped):
      (a) protein-protein < 8 Å and protein-peptide < 14 Å from radius_graph;
      (b) the peptide as a COMPLETE subgraph (any distance) so it is always
          internally connected — even at high diffusion noise;
      (c) each peptide residue to its ``pep_mhc_k`` nearest MHC residues (k-NN,
          any distance) so the peptide always 'feels' the groove.
    Returns (edge_index[2,E], edge_attr[E,5])."""
    posf = pos.float()
    pb = pep.bool()
    n = pos.shape[0]
    n_graphs = int(batch.max()) + 1

    # (a) radius graph, keep pro-pro<8 and mixed pro-pep<14 (pep-pep handled by (b))
    ei = radius_graph(posf, r=cut_pm, batch=batch, loop=False, max_num_neighbors=max_nb)
    d = (posf[ei[1]] - posf[ei[0]]).norm(dim=-1)
    pi, pj = pb[ei[1]], pb[ei[0]]
    mix = pi ^ pj
    keep = ((~pi) & (~pj) & (d < cut_pp)) | (mix & (d < cut_pm))
    ei_rad = ei[:, keep]

    # (b) complete peptide subgraph
    pep_idx = pb.nonzero(as_tuple=False).flatten()
    ei_pep = _intra_group_pairs(pep_idx, batch[pep_idx], n_graphs)

    # (c) peptide <-> nearest MHC (both directions), guaranteed contact
    mhc_idx = (~pb).nonzero(as_tuple=False).flatten()
    if pep_idx.numel() and mhc_idx.numel():
        a = knn(posf[mhc_idx], posf[pep_idx], min(pep_mhc_k, mhc_idx.shape[0]),
                batch[mhc_idx], batch[pep_idx])          # [2, Pep*k]: (pep, mhc)
        pg, mg = pep_idx[a[0]], mhc_idx[a[1]]
        ei_pm = torch.stack([torch.cat([pg, mg]), torch.cat([mg, pg])], 0)
    else:
        ei_pm = pep_idx.new_zeros(2, 0)

    # combine + dedupe directed edges
    edge_index = torch.cat([ei_rad, ei_pep, ei_pm], dim=1)
    key = torch.unique(edge_index[0].long() * n + edge_index[1].long())
    edge_index = torch.stack([key // n, key % n], 0)

    i, j = edge_index[1], edge_index[0]
    pi, pj = pb[i], pb[j]
    same = (pi == pj).float()
    sep = ((residue_index[i] - residue_index[j]).abs().clamp(max=32).float()
           / 32.0) * same
    etype = torch.stack([pi & pj, (~pi) & (~pj), pi ^ pj], -1).float()
    edge_attr = torch.cat([etype, same[:, None], sep[:, None]], -1)   # [E,5]
    return edge_index, edge_attr


class EGNNLayer(nn.Module):
    def __init__(self, h_dim, edge_dim=5, m_dim=128, use_cross=True):
        super().__init__()
        self.use_cross = use_cross
        self.phi_m = mlp([2 * h_dim + 1 + edge_dim, m_dim, m_dim])
        self.phi_x = mlp([m_dim, m_dim, 1], zero_last=True)
        if use_cross:
            self.phi_c = mlp([2 * m_dim, m_dim, 1], zero_last=True)
        self.phi_h = mlp([h_dim + m_dim, m_dim, h_dim])

    def forward(self, h, pos, edge_index, edge_attr, tri):
        i, j = edge_index[1], edge_index[0]
        rij = pos[i] - pos[j]
        d = rij.norm(dim=-1, keepdim=True)
        m = self.phi_m(torch.cat([h[i], h[j], d ** 2, edge_attr], -1))   # [E,m]
        n = h.shape[0]
        # radial (E(3)) coordinate update, mean-aggregated
        contrib = (rij / (d + 1.0)) * self.phi_x(m)
        dx = scatter(contrib, i, dim=0, dim_size=n, reduce="mean")
        # chirality (SE(3)) via k-NN triplets centred at node c
        if self.use_cross and tri is not None:
            c, ea, eb = tri                              # center, edge a, edge b
            ua = rij[ea] / (d[ea] + 1e-6)
            ub = rij[eb] / (d[eb] + 1e-6)
            w = self.phi_c(torch.cat([m[ea], m[eb]], -1))
            dx = dx + scatter(torch.cross(ua, ub, dim=-1) * w, c, dim=0,
                              dim_size=n, reduce="mean")
        m_agg = scatter(m, i, dim=0, dim_size=n, reduce="sum")
        h = h + self.phi_h(torch.cat([h, m_agg], -1))
        return h, pos + dx


def knn_triplets(pos, batch, k=12):
    """Build (center, edge_a, edge_b) triplets from a k-NN graph — every two
    incident edges of each centre node are paired. Fully vectorized (no Python
    loop): all torch ops, O(E + ΣC²) where C = per-node degree (≈ k)."""
    ei = knn_graph(pos.float(), k=k, batch=batch, loop=False,
                   flow="target_to_source")            # torch_cluster needs fp32
    n, dev = pos.shape[0], pos.device
    i = ei[1]                                            # centre (target) per edge
    E = i.shape[0]
    if E == 0:
        return ei, None
    order = torch.argsort(i)                             # group edges by centre
    i_sorted = i[order]
    counts = torch.bincount(i_sorted, minlength=n)       # edges per centre
    starts = torch.cumsum(counts, 0) - counts            # group start (sorted space)
    rep = counts[i_sorted]                               # group size per sorted edge
    gs = starts[i_sorted]                                # group start per sorted edge
    total = int(rep.sum())                               # = ΣC² (pairs incl. self)
    if total == 0:
        return ei, None
    # a = each sorted edge repeated (its group size) times; b = the group's edges
    a_sorted = torch.repeat_interleave(torch.arange(E, device=dev), rep)
    cum = torch.cumsum(rep, 0) - rep
    local = torch.arange(total, device=dev) - torch.repeat_interleave(cum, rep)
    b_sorted = torch.repeat_interleave(gs, rep) + local
    keep = a_sorted != b_sorted                          # drop self-pairs (a==b)
    a_edge, b_edge = order[a_sorted[keep]], order[b_sorted[keep]]
    return ei, (i[a_edge], a_edge, b_edge)               # (centre, edge_a, edge_b)


class EGNNDenoiser(nn.Module):
    def __init__(self, n_aatype, n_segments=4, h_dim=128, m_dim=128, n_layers=6,
                 k=12, pe_dim=32, use_time=True, use_cross=True):
        super().__init__()
        self.k, self.use_time = k, use_time
        self.n_aatype, self.n_segments, self.pe_dim = n_aatype, n_segments, pe_dim
        in_dim = n_aatype + n_segments + pe_dim + (1 if use_time else 0)
        self.embed = mlp([in_dim, h_dim, h_dim])
        self.layers = nn.ModuleList(
            [EGNNLayer(h_dim, m_dim=m_dim, use_cross=use_cross) for _ in range(n_layers)])

    def node_features(self, data, t_frac=None):
        aa = F.one_hot(data.aatype.clamp_min(0), self.n_aatype).float()
        seg = F.one_hot(data.segment_id.clamp_min(0), self.n_segments).float()
        pe = sinusoidal(data.residue_index, self.pe_dim)
        feats = [aa, seg, pe]
        if self.use_time:
            feats.append(t_frac[data.batch][:, None])    # per-graph t -> per node
        return self.embed(torch.cat(feats, -1))

    def forward(self, pos, h0, data):
        edge_index, edge_attr = build_message_edges(
            pos, data.batch, data.pep, data.residue_index)
        _, tri = knn_triplets(pos, data.batch, self.k) if any(
            l.use_cross for l in self.layers) else (None, None)
        x_in, h = pos, h0
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr, tri)
        return pos - x_in, h                             # equivariant ε prediction
