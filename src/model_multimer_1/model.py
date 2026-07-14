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
    StructureModule, InvariantPointAttentionMultimer, AngleResnet)
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


def _frames_from_bb(teacher_bb, seq_mask, noise=0.0, identity_mask=None):
    """Rigid3Array [B,N] backbone frames from teacher_bb [B,N,3,3]=(N,CA,C); optional
    Gaussian noise on the coords for robustness; masked/absent residues -> identity.

    ``identity_mask`` [B,N] bool: additionally force these residues to the identity
    frame. Used to withhold the PEPTIDE's true backbone frames (see MultimerModel
    `pep_frames`) — otherwise the ground-truth peptide pose leaks into the trunk IPAs.
    """
    bb = teacher_bb
    if noise > 0:
        bb = bb + noise * torch.randn_like(bb)
    r = Rigid.from_3_points(bb[..., 0, :], bb[..., 1, :], bb[..., 2, :])        # [B,N]
    t4 = r.to_tensor_4x4()                                                       # [B,N,4,4]
    ident = torch.eye(4, device=t4.device).expand_as(t4)
    keep = seq_mask.bool()
    if identity_mask is not None:
        keep = keep & ~identity_mask.bool()
    t4 = torch.where(keep[..., None, None], t4, ident)
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
def _fp32(fn, *args, **kw):
    """Run `fn` OUTSIDE autocast, in fp32. bf16 keeps only ~3 decimal digits; the trunk's
    N^2 triangle products and the IPA amplify that rounding error into huge SPURIOUS
    gradients (measured: same loss, but max |grad| 2122 in bf16 vs 250 in fp32). Casting
    just these ops to fp32 removes the spikes at a fraction of the cost of full fp32."""
    with torch.autocast("cuda", enabled=False):
        args = tuple(a.float() if torch.is_tensor(a) else a for a in args)
        kw = {k: (v.float() if torch.is_tensor(v) else v) for k, v in kw.items()}
        return fn(*args, **kw)


class TrunkBlock(nn.Module):
    def __init__(self, c_s=D, c_z=D, n_ipa=2, n_heads=IPA_HEADS, p_drop=DROPOUT,
                 fp32_ops=()):
        super().__init__()
        self.fp32_ops = set(fp32_ops)
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

    def _run(self, name, fn, *args, **kw):
        """Run op `name` in fp32 if it is in `fp32_ops`, else at the ambient precision."""
        if name in self.fp32_ops:
            return _fp32(fn, *args, **kw)
        return fn(*args, **kw)

    def forward(self, s, z, frames, mask):
        pair_mask = mask[:, :, None] * mask[:, None, :]
        # ---- pair update (dropout on each additive branch) ----
        z = z + self.drop_z(self._run("opm", self.opm, s[:, None], mask=mask[:, None]))
        z = z + self.drop_z(self._run("tri", self.tri_out, z, mask=pair_mask))
        z = z + self.drop_z(self._run("tri", self.tri_in, z, mask=pair_mask))
        z = z + self.drop_z(self._run("tri", self.pair_trans, z, mask=pair_mask))
        # ---- single update: 2 IPAs (pair+geom -> single) + self-attention ----
        for ipa in self.ipas:
            # The IPA is left at the ambient precision: the ABLATION showed the triangle
            # ops are the culprit, and forcing the IPA to fp32 buys nothing (and its
            # Rigid3Array frames make a clean fp32 cast awkward).
            s = s + self.drop_s(ipa(self.s_norm(s), z, frames, mask))
        key_pad = ~mask.bool()
        s = s + self.self_attn(s, s, s, key_padding_mask=key_pad, need_weights=False)[0]
        return s, z


class MultimerModel(nn.Module):
    """`pep_frames` controls what geometric frames the PEPTIDE residues get:
      "teacher"  — the peptide's TRUE backbone frames (from teacher_bb) are fed to the
                   trunk IPAs. This LEAKS the ground-truth peptide pose: the model can
                   read the answer instead of docking. It is what every checkpoint so
                   far was trained with, so it stays the default for reproducibility.
      "identity" — peptide frames are the identity (the DOCUMENTED design). Only the MHC
                   provides geometry; the peptide pose must actually be predicted.
                   Correct/deployable, but requires retraining.
    """

    def __init__(self, n_trunk=1, mhc_noise=0.5, device="cpu", pep_frames="identity",
                 trunk_fp32=("tri",)):
        super().__init__()
        assert pep_frames in ("teacher", "identity")
        self.pep_frames = pep_frames
        # Ops inside the trunk to run in fp32 (see _fp32). MEASURED on a frozen model
        # over 120 structures (max |grad|, median unchanged at ~2.7 in every case):
        #     all bf16        967.0     <- what killed the 13_08 run
        #     tri in fp32      29.4     <- DEFAULT: 33x smaller spikes
        #     opm in fp32     116.8
        #     everything fp32  68.3     <- slower AND worse than tri-only
        # The N^2 triangle products are where bf16's ~3 decimal digits get amplified into
        # huge SPURIOUS gradients (the loss is fine; the gradient is numerical garbage).
        self.trunk_fp32 = tuple(trunk_fp32 or ())
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
        self.trunk = nn.ModuleList([TrunkBlock(fp32_ops=self.trunk_fp32)
                                    for _ in range(n_trunk)])   # at D
        # ONLY here do we widen from D back to the frozen SM / pLDDT-head dims
        self.sm_s = nn.Linear(D, SM_C_S)
        self.sm_z = nn.Linear(D, SM_C_Z)
        # The frozen pLDDT head was pretrained on the SM's OWN `single`; feeding it a
        # trunk-derived vector is the same out-of-distribution mistake as the frozen
        # angle_resnet. It now reads sm["single"] through a trainable adapter (which is
        # also the only thing stage 3 trains).
        self.plddt_proj = nn.Linear(SM_C_S, SM_C_S)
        # identity init => the frozen pLDDT head sees sm["single"] verbatim, exactly as in
        # AlphaFold. Stage 1 (adapter frozen) therefore reproduces AF's own confidence,
        # and stage 2/3 fine-tune from that instead of from a random projection.
        nn.init.eye_(self.plddt_proj.weight)
        nn.init.zeros_(self.plddt_proj.bias)
        # TRAINABLE torsion head (AF2's own AngleResnet, freshly initialised). The frozen
        # sm.angle_resnet is never used: it maps AF-Evoformer `single` -> torsions and is
        # garbage on our representation (chi1 was 10.8% correct vs a 22% random baseline).
        self.angle_head = AngleResnet(c_in=SM_C_S, c_hidden=128, no_blocks=2,
                                      no_angles=7, epsilon=1e-8)
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
        """Configure trainable params per stage. The StructureModule AND the AF pLDDT
        head are ALWAYS frozen — only the encoder path + torsion head are ever trained.

          1: structure only (lambda_plddt = 0). embedder + head-2 + trunk + projections
             + angle_head trainable. `plddt_proj` is FROZEN: with a zero pLDDT weight it
             would receive no gradient, i.e. be a silent unused parameter (and force
             DDP find_unused_parameters=True).
          2: same PLUS `plddt_proj` trainable (lambda_plddt = 0.01).
          3: freeze EVERYTHING except `plddt_proj`; the forward detaches the SM single
             before it, so confidence learning cannot move the structure.
        """
        self.requires_grad_(True)                        # reset, then re-freeze below
        self.detach_plddt = False
        if stage == 1:
            self.plddt_proj.requires_grad_(False)        # no gradient at lambda_plddt=0
        elif stage == 3:
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

    def _encode(self, batch):
        """seq + MHC geometry + anchors -> (single[B,N,D], pair[B,N,N,D], mask)."""
        f = build_multimer_feats(batch["aatype"], batch["segment_id"],
                                 batch["seq_mask"])
        # head-1
        msa_emb, z = self.embedder(f)                    # [B,1,N,c_m], [B,N,N,c_z]
        s = msa_emb[..., 0, :, :]                          # [B,N,c_m]
        # head-2 (noised MHC frames). pep_frames="identity" withholds the peptide's
        # true backbone frames so the pose is predicted, not read off the input.
        idm = (m1.peptide_mask_from_batch(batch["seq_mask"], batch["segment_id"])
               if self.pep_frames == "identity" else None)
        frames = _frames_from_bb(batch["teacher_bb"], batch["seq_mask"],
                                 noise=self.mhc_noise if self.training else 0.0,
                                 identity_mask=idm)
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
        return s, z, mask

    def _structure(self, batch):
        """Encoder -> frozen SM (backbone) -> TRAINABLE torsion head -> full-atom14.

        The SM's own angle_resnet is bypassed. Side-chain atoms are rebuilt from OUR
        torsions with the SM's `torsion_angles_to_frames` /
        `frames_and_literature_positions_to_atom14_pos`, which hold **no learned
        parameters** (only the `default_frames`/`group_idx`/`lit_positions` buffers) —
        so the StructureModule stays entirely frozen, exactly as AlphaFold's geometry.
        """
        s, z, mask = self._encode(batch)
        sm_in = self.sm_s(s)
        out = self.sm({"single": sm_in, "pair": self.sm_z(z)},
                      batch["aatype"], mask=mask)
        sm_single = out["single"]                                     # [B,N,384]
        # torsions: (omega, phi, psi, chi1..chi4); unnormalized feeds the angle-norm reg
        unnorm_angles, angles = self.angle_head(sm_single, sm_in)     # [B,N,7,2] each
        # rebuild all-atom from the SM's final backbone frames + OUR torsions. The
        # multimer SM emits `frames` as a Rigid3Array 4x4 array [layers,B,N,4,4],
        # already scale_translation'd into Å (same rigid it feeds its own angle path).
        bb_rigid = Rigid3Array.from_array4x4(out["frames"][-1])
        all_frames = self.sm.torsion_angles_to_frames(bb_rigid, angles, batch["aatype"])
        atom14 = self.sm.frames_and_literature_positions_to_atom14_pos(
            all_frames, batch["aatype"])                              # [B,N,14,3]
        return out, sm_single, unnorm_angles, angles, all_frames, atom14, mask

    def predict(self, batch):
        """Full-atom inference: atom14 (backbone from the frozen SM, side chains from the
        trained torsion head) + pLDDT logits. No grad; caller should be in eval()."""
        with torch.no_grad():
            out, sm_single, _, _, _, atom14, _ = self._structure(batch)
            return {"atom14": atom14,                       # [B,N,14,3]
                    "ca": atom14[..., 1, :],                # [B,N,3]
                    "plddt_logits": self.plddt(self.plddt_proj(sm_single))}

    def forward(self, batch, return_frames=False, num_recycles=None):
        out, sm_single, unnorm_angles, angles, all_frames, atom14, mask = \
            self._structure(batch)
        ca = atom14[..., 1, :]                    # CA sits in rigid-group 0 (backbone)
        # stage-3: confidence only -> no gradient may reach the structure
        s_plddt = sm_single.detach() if self.detach_plddt else sm_single
        plddt_logits = self.plddt(self.plddt_proj(s_plddt))
        # PAE is never trained here (teacher PAE is zeros for hasmig; lambda_pae=0).
        # Returning None instead of a [B,N,N,64] zero tensor saves ~37 MB and an N^2x64
        # cross-entropy per example; DistillLoss/eval_metrics treat None as "no PAE".
        pae_logits = None
        if return_frames:
            aux = {"angles": angles, "unnormalized_angles": unnorm_angles,
                   "sidechain_frames": all_frames.to_tensor_4x4(),   # [B,N,8,4,4]
                   "atom14": atom14}
            return ca, plddt_logits, pae_logits, out["frames"], aux
        return ca, plddt_logits, pae_logits


def _dummy_batch(dev, B=1, n_mhc=180, n_pep=9):
    N = n_mhc + n_pep
    batch = {
        "aatype": torch.randint(0, 20, (B, N), device=dev),
        "segment_id": torch.tensor([[0] * n_mhc + [1] * n_pep] * B, device=dev),
        "seq_mask": torch.ones(B, N, device=dev),
        "anchor": (torch.rand(B, N, device=dev) > 0.8).float(),
        "teacher_bb": torch.randn(B, N, 3, 3, device=dev) * 10,
        "teacher_ca": torch.randn(B, N, 3, device=dev) * 10,
    }
    batch["teacher_plddt"] = torch.rand(B, N, device=dev) * 100
    batch["teacher_pae"] = torch.zeros(B, N, N, device=dev)
    # AF2 sidechain targets, so the smoke exercises sc-FAPE + chi (not a silent no-op)
    from openfold.np import residue_constants as rc
    aa = batch["aatype"]
    a14m = torch.tensor(rc.restype_atom14_mask, device=dev, dtype=torch.float32)[aa]
    chim = torch.tensor(rc.chi_angles_mask, device=dev, dtype=torch.float32)[aa]
    batch["teacher_atom14"] = torch.randn(B, N, 14, 3, device=dev) * 5 * a14m[..., None]
    batch["teacher_atom14_mask"] = a14m
    chi = torch.randn(B, N, 4, 2, device=dev)
    batch["teacher_chi"] = chi / chi.norm(dim=-1, keepdim=True)   # unit (sin,cos)
    batch["teacher_chi_mask"] = chim
    return batch


def _leak_check(dev=None, ckpt=None, tol=1e-6):
    """REGRESSION GUARD for the peptide-pose leak.

    `frames` is the ONLY channel through which teacher_bb reaches the peptide (head-2's
    pair/IPA are masked to the MHC block), so the guard asserts on the frames directly.
    That is exact and weight-independent — unlike an end-to-end check on a fresh model,
    which is VACUOUS: OpenFold's IPA zero-inits its output projection, so at init the
    frames have no effect on the prediction at all and any leak would go unnoticed.

      pep_frames="identity" -> peptide frames are exactly the identity AND do not move
                               when the peptide's teacher_bb is perturbed.
      pep_frames="teacher"  -> peptide frames DO move (positive control: proves the
                               test can actually detect a leak).

    Pass ``ckpt`` (a TRAINED checkpoint) to additionally assert end-to-end that the
    prediction itself is invariant to the peptide's teacher_bb.
    """
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    batch = _dummy_batch(dev)
    pep = m1.peptide_mask_from_batch(batch["seq_mask"], batch["segment_id"]).bool()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["teacher_bb"][pep] += torch.tensor([10.0, 0.0, 0.0], device=dev)

    def pep_frames_t4(bb, idm):
        return _frames_from_bb(bb, batch["seq_mask"], 0.0, idm).to_tensor_4x4()[pep]

    # --- pep_frames="identity": frames are identity and invariant to the peptide bb
    f0 = pep_frames_t4(batch["teacher_bb"], pep)
    f1 = pep_frames_t4(b2["teacher_bb"], pep)
    d_id = (f1 - f0).abs().max().item()
    off_ident = (f0 - torch.eye(4, device=dev).expand_as(f0)).abs().max().item()
    # --- positive control: pep_frames="teacher" DOES follow the perturbation
    d_te = (pep_frames_t4(b2["teacher_bb"], None)
            - pep_frames_t4(batch["teacher_bb"], None)).abs().max().item()

    print(f"leak-check: peptide teacher_bb +10A -> max|Δ peptide frame|  "
          f"identity={d_id:.2e}  teacher={d_te:.3f} | identity-frame err={off_ident:.2e}")
    assert off_ident < tol, f"pep_frames='identity' frames are not the identity ({off_ident:.2e})"
    assert d_id < tol, (
        f"PEPTIDE POSE LEAK: under pep_frames='identity' the peptide frames moved "
        f"{d_id:.3e} when only the peptide's teacher_bb changed — ground-truth peptide "
        f"geometry is reaching the trunk.")
    assert d_te > 1.0, ("positive control failed: pep_frames='teacher' should follow the "
                        "perturbed input; this test can no longer detect a leak.")

    if ckpt is not None:                       # end-to-end, needs TRAINED weights
        net = MultimerModel(device=dev, pep_frames="identity").eval()
        net.load_state_dict(torch.load(ckpt, map_location=dev,
                                       weights_only=False)["trainable"], strict=False)
        d = (net.predict(b2)["ca"] - net.predict(batch)["ca"])[pep].norm(dim=-1).mean().item()
        print(f"leak-check (end-to-end, {Path(ckpt).name}): |Δ pred peptide Cα| = {d:.2e} A")
        assert d < 1e-4, f"PEPTIDE POSE LEAK end-to-end: prediction moved {d:.4f} A"

    # --- the SIDE-CHAIN TARGETS are loss-only: they must never reach the model --------
    net = MultimerModel(device=dev, pep_frames="identity").eval()
    base = net.predict(batch)["atom14"]
    for key in ("teacher_atom14", "teacher_chi", "teacher_ca", "teacher_plddt"):
        b3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        b3[key] = torch.randn_like(b3[key]) * 10.0          # destroy the target
        d = (net.predict(b3)["atom14"] - base).abs().max().item()
        assert d == 0.0, (f"TARGET LEAK: randomising batch['{key}'] moved the prediction "
                          f"by {d:.3e} A. It is a supervision target, not an input.")
    print("OK leak-check: sidechain/confidence TARGETS never influence the prediction")
    print("OK leak-check: peptide geometry is withheld under pep_frames='identity'")


def _smoke():
    """Build the model + one forward on a synthetic pMHC (run on the cluster)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _leak_check(dev)
    net = MultimerModel(device=dev)
    batch = _dummy_batch(dev)
    net.set_stage(1)
    net.train()
    ca, pl, pae, fr, aux = net(batch, return_frames=True)
    print("OK forward:", tuple(ca.shape), tuple(pl.shape), "pae=None" if pae is None else
          tuple(pae.shape), "| atom14:", tuple(aux["atom14"].shape),
          "| trainable:", sum(p.numel() for p in net.trainable_parameters()))
    # ---- LOSS + BACKWARD, with AF2's side-chain terms actually ON ----
    loss_mod = m1.DistillLoss(0.5, 0.0, 0.0, peptide_weight=5.0,
                              lambda_sc_fape=0.5, lambda_chi=1.0).to(dev)
    total, terms = loss_mod(ca, pl, pae, fr, batch, aux=aux)
    assert float(terms["sc_fape"]) > 0 and float(terms["chi_loss"]) > 0, \
        "side-chain losses are zero -- they are a silent no-op"
    total.backward()
    n_grad = sum(1 for p in net.trainable_parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all())
    n_tot = sum(1 for _ in net.trainable_parameters())
    ah = sum(p.grad.norm().item() ** 2 for p in net.angle_head.parameters()
             if p.grad is not None) ** 0.5
    assert ah > 0, "no gradient reaches the torsion head"
    assert not any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in net.sm.parameters()), "StructureModule received gradient"
    print(f"OK loss: total={float(total):.3f} fape={terms['fape']:.3f} "
          f"sc_fape={terms['sc_fape']:.3f} chi={terms['chi_loss']:.3f} "
          f"finite={torch.isfinite(total).item()} | grads {n_grad}/{n_tot} "
          f"| ||grad angle_head||={ah:.3e} | SM grads: 0")


if __name__ == "__main__":
    _smoke()
