"""
model_multimer_1 — a slim, two-head model on top of the FROZEN AlphaFold-Multimer
structure module + pLDDT head.

Pipeline (see README):
  head-1 : AF-multimer InputEmbedder on the peptide+MHC sequences (single-seq MSA,
           per-chain residue_index, asym/entity/sym) -> s_seq[N,c_m], z_seq[N,N,c_z]
  head-2 : MHC backbone (+ Gaussian noise) -> template-style structural features
           (distogram + relative-orientation pair feats + backbone-torsion single
           feats) -> 1 multimer IPA -> Linear -> 10-d per-MHC-residue; padded to N.
           pair gets the 10-d (broadcast); single gets its mean-pool.
  anchors: 2-d one-hot (peptide anchor / not) appended to single and pair.
  trunk  : project to (c_s=384, c_z=128); then 2x [ multimer IPA -> single ;
           outer-product-mean single->pair ] with single self-attention; frames are
           fixed context (MHC noised frames + peptide identity).
  out    : single -> proj -> FROZEN pLDDT head (confidence, decoupled from structure)
           (single,pair) -> FROZEN multimer StructureModule -> Ca + frames (FAPE)

forward returns (ca, plddt_logits, pae_logits(zeros), frames) — the SAME contract as
DistillModel, so it reuses model_1's DistillLoss (lambda_pae=0) + train loop.

NOTE: this wires several OpenFold-multimer internals (multimer IPA needs Rigid3Array
frames; StructureModule(is_multimer=True); InputEmbedderMultimer I/O). It compiles,
but the FIRST run must be the `--smoke` test on the cluster to validate the wiring
(feature dims are confirmed; frame construction + module I/O need a runtime check).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1]
_ROOT = _SRC.parent
sys.path.insert(0, str(_ROOT / "openfold"))
sys.path.insert(0, str(_SRC / "model"))
import utils as m1                                                    # noqa: E402
from openfold.config import model_config                              # noqa: E402
from openfold.model.embedders import InputEmbedderMultimer            # noqa: E402
from openfold.model.structure_module import (                         # noqa: E402
    StructureModule, InvariantPointAttentionMultimer)
from openfold.model.heads import PerResidueLDDTCaPredictor            # noqa: E402
from openfold.model.outer_product_mean import OuterProductMean        # noqa: E402
from openfold.model.triangular_multiplicative_update import (         # noqa: E402
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
from openfold.model.pair_transition import PairTransition             # noqa: E402
from openfold.utils.rigid_utils import Rigid                          # noqa: E402
from openfold.utils.geometry.rigid_matrix_vector import Rigid3Array   # noqa: E402

PARAMS = _ROOT / "params" / "alphafold"
MM = "model_1_multimer_v3"
# pretrained AF-multimer widths (fixed by the loaded embedder / SM / pLDDT head):
C_M, C_Z_EMB = 256, 128              # InputEmbedderMultimer single / pair outputs
SM_C_S, SM_C_Z = 384, 128           # frozen StructureModule + pLDDT head single / pair
# our SMALL internal working width for head-2 / the trunk (single AND pair); we only
# widen back to the SM/pLDDT dims in the final projections.
D = 64
IPA_HEADS = 4
DROPOUT = 0.1                        # applied on the TRAINABLE layers for robustness


# --------------------------------------------------------------------------- #
# our padded [B,N] batch  ->  multimer InputEmbedder inputs
# --------------------------------------------------------------------------- #
def build_multimer_feats(aatype, segment_id, seq_mask):
    """target_feat[B,N,21], msa_feat[B,1,N,49], per-CHAIN residue_index, asym/entity/
    sym ids, masks — the inputs InputEmbedderMultimer.forward consumes."""
    B, N = aatype.shape
    dev = aatype.device
    m = seq_mask.bool()
    pep = (segment_id == segment_id.max(dim=1, keepdim=True).values) & m       # [B,N]
    mhc = m & ~pep
    aa = aatype.clamp(0, 20).long()
    tf = F.one_hot(aa, 21).float()
    oh23 = F.one_hot(aa, 23).float()[:, None]                                  # [B,1,N,23]
    zero = torch.zeros(B, 1, N, 1, device=dev)
    msa = torch.cat([oh23, zero, zero, oh23, zero], -1)                         # [B,1,N,49]
    # per-chain residue_index (each chain numbered from 0); asym/entity/sym ids
    ri = torch.zeros(B, N, dtype=torch.long, device=dev)
    asym = torch.zeros(B, N, dtype=torch.long, device=dev)
    for b in range(B):
        mi = mhc[b].nonzero(as_tuple=True)[0]
        pi = pep[b].nonzero(as_tuple=True)[0]
        ri[b, mi] = torch.arange(len(mi), device=dev)
        ri[b, pi] = torch.arange(len(pi), device=dev)
        asym[b, mi] = 1
        asym[b, pi] = 2
    entity = asym.clone()                    # MHC & peptide are distinct sequences
    sym = m.long()                           # single copy of each -> sym_id 1
    return {"target_feat": tf, "msa_feat": msa, "residue_index": ri,
            "asym_id": asym, "entity_id": entity, "sym_id": sym,
            "seq_mask": seq_mask, "pep": pep.float(), "mhc": mhc.float()}


def _frames_from_bb(teacher_bb, seq_mask, noise=0.0):
    """Rigid3Array [B,N] backbone frames from teacher_bb [B,N,3,3]=(N,CA,C); optional
    Gaussian noise on the coords for robustness; masked/absent residues -> identity."""
    bb = teacher_bb
    if noise > 0:
        bb = bb + noise * torch.randn_like(bb)
    r = Rigid.from_3_points(bb[..., 0, :], bb[..., 1, :], bb[..., 2, :])        # [B,N]
    t4 = r.to_tensor_4x4()                                                       # [B,N,4,4]
    ident = torch.eye(4, device=t4.device).expand_as(t4)
    t4 = torch.where(seq_mask[..., None, None].bool(), t4, ident)
    return Rigid3Array.from_array4x4(t4)


# --------------------------------------------------------------------------- #
# head-2: template-style MHC structure encoder
# --------------------------------------------------------------------------- #
class MHCStructEncoder(nn.Module):
    """MHC backbone -> distogram + relative-orientation pair feats + torsion single
    feats -> 1 multimer IPA -> Linear -> 10-d per residue."""

    def __init__(self, d_out=10, n_dist_bins=22, c_s=D, c_z=D, p_drop=DROPOUT):
        super().__init__()
        self.n_dist_bins = n_dist_bins
        self.register_buffer("dist_bins", torch.linspace(3.0, 22.0, n_dist_bins - 1))
        # pair feats: distogram + 3 relative-orientation unit-vector comps
        self.pair_in = nn.Linear(n_dist_bins + 3, c_z)
        self.single_in = nn.Linear(3, c_s)                 # 3 torsion-ish placeholders
        self.ipa = InvariantPointAttentionMultimer(
            c_s=c_s, c_z=c_z, c_hidden=16, no_heads=IPA_HEADS,
            no_qk_points=4, no_v_points=8)
        self.drop = nn.Dropout(p_drop)
        self.out = nn.Linear(c_s, d_out)

    def forward(self, teacher_bb, mhc_mask, frames):
        """teacher_bb[B,N,3,3], mhc_mask[B,N], frames Rigid3Array[B,N]. Returns
        (h2_single[B,N,10], h2_pair[B,N,N,25]) — both nonzero only on the MHC block.
        h2_single is the per-residue IPA summary; h2_pair is the raw distogram +
        relative-orientation geometry, kept so it can be injected into the trunk pair."""
        ca = teacher_bb[..., 1, :]                                    # [B,N,3]
        d = torch.cdist(ca, ca)                                       # [B,N,N]
        dist = F.one_hot(torch.bucketize(d, self.dist_bins),
                         self.n_dist_bins).float()
        # relative orientation: direction to j in i's local frame (unit vector)
        rel = frames[..., None].invert_apply(ca[..., None, :, :])     # [B,N,N,3]
        rel = rel / (rel.norm(dim=-1, keepdim=True) + 1e-6)
        feat = torch.cat([dist, rel], -1)                            # [B,N,N,25]
        mhc_pair = (mhc_mask[:, :, None] * mhc_mask[:, None, :])[..., None]
        feat = feat * mhc_pair                                        # MHC×MHC block only
        z = self.pair_in(feat)                                        # [B,N,N,c_z] (IPA bias)
        s = self.single_in(ca * 0.0)  # placeholder torsion feats (zeros) -> c_s
        upd = self.ipa(s, z, frames, mhc_mask)                        # [B,N,c_s]
        h2_single = self.out(self.drop(upd)) * mhc_mask[..., None]    # [B,N,10]
        return h2_single, feat                                        # feat=[B,N,N,25]


# --------------------------------------------------------------------------- #
# trunk block: 2x (IPA -> single ; OPM single->pair) + single self-attention
# --------------------------------------------------------------------------- #
class TrunkBlock(nn.Module):
    def __init__(self, c_s=D, c_z=D, n_ipa=2, n_heads=IPA_HEADS, p_drop=DROPOUT):
        super().__init__()
        self.ipas = nn.ModuleList([
            InvariantPointAttentionMultimer(c_s=c_s, c_z=c_z, c_hidden=16,
                                            no_heads=IPA_HEADS, no_qk_points=4,
                                            no_v_points=8)
            for _ in range(n_ipa)])
        # PAIR update path: OPM (single->pair) + triangle mult (pair->pair) + transition
        self.opm = OuterProductMean(c_s, c_z, c_hidden=32)
        self.tri_out = TriangleMultiplicationOutgoing(c_z, c_hidden=c_z)
        self.tri_in = TriangleMultiplicationIncoming(c_z, c_hidden=c_z)
        self.pair_trans = PairTransition(c_z, n=2)
        self.s_norm = nn.LayerNorm(c_s)
        self.self_attn = nn.MultiheadAttention(c_s, n_heads, dropout=p_drop,
                                               batch_first=True)
        # dropout on each residual branch (single & pair) for regularization
        self.drop_s = nn.Dropout(p_drop)
        self.drop_z = nn.Dropout(p_drop)

    def forward(self, s, z, frames, mask):
        pair_mask = mask[:, :, None] * mask[:, None, :]
        # ---- pair update (dropout on each additive branch) ----
        z = z + self.drop_z(self.opm(s[:, None], mask=mask[:, None]))  # single -> pair
        z = z + self.drop_z(self.tri_out(z, mask=pair_mask))          # triangle (outgoing)
        z = z + self.drop_z(self.tri_in(z, mask=pair_mask))           # triangle (incoming)
        z = z + self.drop_z(self.pair_trans(z, mask=pair_mask))       # pair feed-forward
        # ---- single update: 2 IPAs (pair+geom -> single) + self-attention ----
        for ipa in self.ipas:
            s = s + self.drop_s(ipa(self.s_norm(s), z, frames, mask))
        key_pad = ~mask.bool()
        s = s + self.self_attn(s, s, s, key_padding_mask=key_pad, need_weights=False)[0]
        return s, z


class MultimerModel(nn.Module):
    def __init__(self, n_trunk=1, mhc_noise=0.5, device="cpu"):
        super().__init__()
        cfg = model_config(MM)
        ie = cfg["model"]["input_embedder"]
        self.embedder = InputEmbedderMultimer(
            tf_dim=21, msa_dim=49, c_z=C_Z_EMB, c_m=C_M,
            max_relative_idx=ie["max_relative_idx"],
            use_chain_relative=ie["use_chain_relative"],
            max_relative_chain=ie["max_relative_chain"])
        self.head2 = MHCStructEncoder()                    # works at D
        self.mhc_noise = mhc_noise
        # single: embedder 256 + head-2 pooled-summary 10 + anchor 2 -> D
        # pair  : embedder 128 + head-2 geometry 25 + anchor 2 -> D
        self.s_proj = nn.Linear(C_M + 10 + 2, D)
        self.z_proj = nn.Linear(C_Z_EMB + 25 + 2, D)
        self.drop_s = nn.Dropout(DROPOUT)                  # on the trunk-input projections
        self.drop_z = nn.Dropout(DROPOUT)
        self.trunk = nn.ModuleList([TrunkBlock() for _ in range(n_trunk)])  # at D
        # ONLY here do we widen from D back to the frozen SM / pLDDT-head dims
        self.sm_s = nn.Linear(D, SM_C_S)
        self.sm_z = nn.Linear(D, SM_C_Z)
        self.plddt_proj = nn.Linear(D, SM_C_S)
        # frozen AF-multimer structure module + pLDDT head
        sm_cfg = {**dict(cfg["model"]["structure_module"]), "is_multimer": True}
        self.sm = StructureModule(**sm_cfg)
        self.plddt = PerResidueLDDTCaPredictor(**cfg["model"]["heads"]["lddt"])
        self._load_frozen()
        self.recycles = 0
        self.detach_plddt = False
        self.to(device)
        self.train(False)

    def _load_frozen(self):
        self.embedder.load_state_dict(torch.load(PARAMS / "input_embedder_mm.pt",
                                                 weights_only=True))
        self.sm.load_state_dict(torch.load(PARAMS / "sm_mm.pt", weights_only=True))
        self.plddt.load_state_dict(torch.load(PARAMS / "plddt_mm.pt", weights_only=True))
        # NB: our IPAs run at D=64 / 4 heads, so they can't be seeded from the SM's
        # 384/12-head multimer-IPA weights — they train from scratch. (The embedder,
        # SM and pLDDT head are the pretrained pieces.)
        for mod in (self.sm, self.plddt):                # frozen structure + confidence
            mod.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sm.eval()
        self.plddt.eval()
        return self

    def set_stage(self, stage: int):
        """Configure trainable params per stage. The StructureModule AND the pLDDT
        head are ALWAYS frozen — only the encoder path is ever trained.
          1 & 2: embedder + head-2 + trunk + projections trainable (SM + pLDDT frozen).
                 Stages differ by DATA/loss regime, not by what is unfrozen.
          3: freeze EVERYTHING except the pLDDT projection; the forward also detaches
             the trunk output before it, so the structure can't move (confidence only)."""
        if stage == 3:
            self.requires_grad_(False)
            self.plddt_proj.requires_grad_(True)
            self.detach_plddt = True
        self.sm.requires_grad_(False)                    # StructureModule ALWAYS frozen
        self.plddt.requires_grad_(False)                 # AF pLDDT head ALWAYS frozen

    def unfreeze_sm_pct(self, pct: float):
        """Unfreeze the LAST `pct`% of the StructureModule's parameters — a light-touch
        fine-tune of the (otherwise frozen) structure decoder, e.g. after N warmup
        epochs of trunk-only training. The pLDDT head stays frozen. Params are counted
        from the last tensor backward (closest to the output). Returns
        (unfrozen_params, total_sm_params)."""
        sm_params = list(self.sm.parameters())
        total = sum(p.numel() for p in sm_params)
        target = total * pct / 100.0
        unfrozen = 0
        for p in reversed(sm_params):
            if unfrozen >= target:
                break
            p.requires_grad_(True)
            unfrozen += p.numel()
        return unfrozen, total

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters()
                if p.requires_grad}

    def forward(self, batch, return_frames=False, num_recycles=None):
        f = build_multimer_feats(batch["aatype"], batch["segment_id"],
                                 batch["seq_mask"])
        # head-1
        msa_emb, z = self.embedder(f)                    # [B,1,N,c_m], [B,N,N,c_z]
        s = msa_emb[..., 0, :, :]                          # [B,N,c_m]
        # head-2 (noised MHC frames)
        frames = _frames_from_bb(batch["teacher_bb"], batch["seq_mask"],
                                 noise=self.mhc_noise if self.training else 0.0)
        h2_single, h2_pair = self.head2(batch["teacher_bb"], f["mhc"], frames)  # [B,N,10],[B,N,N,25]
        h2_pool = (h2_single * f["mhc"][..., None]).sum(1, keepdim=True) \
            / f["mhc"].sum(1, keepdim=True).clamp_min(1.0)[..., None]  # meanpool [B,1,10]
        h2_pool = h2_pool.expand(-1, s.shape[1], -1)
        anc = F.one_hot(batch["anchor"].clamp(0, 1).long(), 2).float()  # [B,N,2]
        anc_pair = torch.maximum(anc[:, :, None], anc[:, None, :])       # [B,N,N,2]
        # concat + project down to D: single gets the pooled head-2 summary; pair gets
        # the raw head-2 distogram/orientation geometry.
        s = self.drop_s(self.s_proj(torch.cat([s, h2_pool, anc], -1)))   # [B,N,D]
        z = self.drop_z(self.z_proj(torch.cat([z, h2_pair, anc_pair], -1)))  # [B,N,N,D]
        # trunk (frames as fixed geometric context)
        mask = batch["seq_mask"]
        for blk in self.trunk:
            s, z = blk(s, z, frames, mask)
        # frozen heads
        out = self.sm({"single": self.sm_s(s), "pair": self.sm_z(z)},
                      batch["aatype"], mask=mask)
        ca = out["positions"][-1][..., 1, :]
        s_plddt = s.detach() if self.detach_plddt else s   # stage-3: confidence only
        plddt_logits = self.plddt(self.plddt_proj(s_plddt))
        B, N = mask.shape
        pae_logits = ca.new_zeros(B, N, N, 64)           # no PAE; DistillLoss lambda_pae=0
        if return_frames:
            return ca, plddt_logits, pae_logits, out["frames"]
        return ca, plddt_logits, pae_logits


def _smoke():
    """Build the model + one forward on a synthetic pMHC (run on the cluster)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = MultimerModel(device=dev)
    B, n_mhc, n_pep = 1, 180, 9
    N = n_mhc + n_pep
    seg = torch.tensor([[0] * n_mhc + [1] * n_pep], device=dev)
    batch = {
        "aatype": torch.randint(0, 20, (B, N), device=dev),
        "segment_id": seg,
        "seq_mask": torch.ones(B, N, device=dev),
        "anchor": (torch.rand(B, N, device=dev) > 0.8).float(),
        "teacher_bb": torch.randn(B, N, 3, 3, device=dev) * 10,
        "teacher_ca": torch.randn(B, N, 3, device=dev) * 10,
    }
    batch["teacher_plddt"] = torch.rand(B, N, device=dev) * 100
    batch["teacher_pae"] = torch.zeros(B, N, N, device=dev)
    net.train()
    ca, pl, pae, fr = net(batch, return_frames=True)
    print("OK forward:", tuple(ca.shape), tuple(pl.shape), tuple(pae.shape),
          "| trainable params:", sum(p.numel() for p in net.trainable_parameters()))
    # ---- validate the LOSS + BACKWARD path (FAPE on the multimer SM frames) ----
    loss_mod = m1.DistillLoss(1.0, 0.01, 0.0, peptide_weight=5.0).to(dev)
    total, terms = loss_mod(ca, pl, pae, fr, batch)
    total.backward()
    n_grad = sum(1 for p in net.trainable_parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all())
    n_tot = sum(1 for _ in net.trainable_parameters())
    print(f"OK loss: total={float(total):.3f} fape={terms['fape']:.3f} "
          f"plddt_ce={terms['plddt_ce']:.3f} finite={torch.isfinite(total).item()} "
          f"| grads on {n_grad}/{n_tot} trainable tensors")


if __name__ == "__main__":
    _smoke()
