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
from torch_cluster import knn_graph, radius_graph
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


def build_message_edges(pos, batch, pep, residue_index,
                        cut_pp=8.0, cut_pm=14.0, max_nb=48
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """radius_graph at the loose 14 Å cutoff, then drop protein-protein pairs
    beyond 8 Å. Returns (edge_index[2,E], edge_attr[E,5])."""
    ei = radius_graph(pos, r=cut_pm, batch=batch, loop=False, max_num_neighbors=max_nb)
    i, j = ei[1], ei[0]                                  # target, source
    d = (pos[i] - pos[j]).norm(dim=-1)
    pep_i, pep_j = pep[i].bool(), pep[j].bool()
    pep_pair = pep_i & pep_j
    pro_pair = (~pep_i) & (~pep_j)
    mix = ~(pep_pair | pro_pair)
    keep = pep_pair | (pro_pair & (d < cut_pp)) | (mix & (d < cut_pm))
    ei = ei[:, keep]
    i, j = ei[1], ei[0]
    pep_i, pep_j = pep[i].bool(), pep[j].bool()
    same = (pep_i == pep_j).float()
    sep = ((residue_index[i] - residue_index[j]).abs().clamp(max=32).float()
           / 32.0) * same
    etype = torch.stack([pep[i].bool() & pep[j].bool(),
                         (~pep[i].bool()) & (~pep[j].bool()),
                         pep[i].bool() ^ pep[j].bool()], -1).float()
    edge_attr = torch.cat([etype, same[:, None], sep[:, None]], -1)   # [E,5]
    return ei, edge_attr


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
    """Build (center, edge_a, edge_b) triplets from a k-NN graph: for each node
    its incident edges are paired. Bounded to O(N·k²)."""
    ei = knn_graph(pos, k=k, batch=batch, loop=False, flow="target_to_source")
    i = ei[1]                                            # target (center) per edge
    order = torch.argsort(i)
    i_sorted = i[order]
    counts = torch.bincount(i_sorted, minlength=pos.shape[0])
    # pair every two edges that share the same center
    starts = torch.cumsum(counts, 0) - counts
    a_list, b_list, c_list = [], [], []
    for node in torch.nonzero(counts >= 2, as_tuple=False).flatten().tolist():
        edges = order[starts[node]:starts[node] + counts[node]]
        aa, bb = torch.meshgrid(edges, edges, indexing="ij")
        mask = aa != bb
        a_list.append(aa[mask]); b_list.append(bb[mask])
        c_list.append(torch.full((int(mask.sum()),), node, device=pos.device))
    if not c_list:
        return ei, None
    return ei, (torch.cat(c_list), torch.cat(a_list), torch.cat(b_list))


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
