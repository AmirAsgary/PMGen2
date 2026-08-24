"""Are the confidence results REAL? Three independent checks.

A high predicted-vs-true pLDDT correlation is exactly what a leak would look like, so it
deserves adversarial verification rather than celebration.

  1. CONFIDENCE LEAK — randomise every teacher_* field on REAL structures and require the
     pLDDT LOGITS to be bit-identical. `_leak_check` historically asserted only on atom14,
     so a leak into the confidence head would have passed unnoticed.
  2. SPLIT INTEGRITY — no base_id (allele+peptide) shared between train and val, and the
     val ids used for scoring are genuinely from the held-out two_axis fold.
  3. UNTRAINED CONTROL — the same correlation measured on a random-init model. If a random
     model also scores well, the metric is measuring something structural about the data
     (e.g. residue position) rather than learned confidence.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "openfold"))
sys.path.insert(0, str(_HERE.parents[1] / "src" / "model"))
sys.path.insert(0, str(_HERE))
import train as T                                                     # noqa: E402
assert "model_multimer_1" in T.__file__
import model as MM                                                    # noqa: E402
import utils as m1                                                    # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_mm1/stage2_snapshot.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"

args = T.parse_args(["--stage", "1"]); args.sidechains = True
args.h5_dir = "data/processed/h5_store_sc"; args.hasmig_dir = "data/processed/h5_store_hasmig"
args.max_val = 100
train_ds, val_full, val_hi = T.build_datasets(args, filt=True, hasmig_w=1.0)

def build(sd=None):
    kw = dict(attn_norm=any("attn_norm" in k for k in sd) if sd else True,
              plddt_adapter=any("plddt_proj" in k for k in sd) if sd else False)
    n = MM.MultimerModel(n_trunk=3, device=dev, pep_frames="identity",
                         angle_input="layernorm", **kw)
    n.set_stage(1)
    if sd: n.load_state_dict(sd, strict=False)
    return n.eval()

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["trainable"]
net = build(sd)

print("="*78); print("1. CONFIDENCE LEAK CHECK on REAL structures")
ld = m1.make_dataloader(val_full, 1, shuffle=False, num_workers=2)
worst = {}
with torch.no_grad():
    for i, b in enumerate(ld):
        if i >= 25: break
        b = m1.move_batch(b, dev)
        base = net.predict(b)
        for key in ("teacher_plddt", "teacher_atom14", "teacher_chi", "teacher_ca",
                    "teacher_pae", "teacher_atom14_mask", "teacher_chi_mask"):
            if key not in b or not torch.is_tensor(b[key]): continue
            b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
            b2[key] = torch.randn_like(b2[key].float()).to(b2[key].dtype) * 10.0
            o2 = net.predict(b2)
            dp = float((o2["plddt_logits"] - base["plddt_logits"]).abs().max())
            da = float((o2["atom14"] - base["atom14"]).abs().max())
            worst[key] = max(worst.get(key, 0.0), max(dp, da))
for k, v in sorted(worst.items()):
    print(f"   randomise {k:22} -> max |Δ pLDDT logits, Δ atom14| = {v:.3e}  {'OK' if v==0 else '*** LEAK ***'}")
print("   VERDICT:", "no leakage" if max(worst.values())==0 else "*** LEAKAGE DETECTED ***")

print("="*78); print("2. SPLIT INTEGRITY")
def ids_of(ds):
    out=[]
    stack=[ds]
    while stack:
        d=stack.pop()
        if hasattr(d,"datasets"): stack.extend(d.datasets)
        elif hasattr(d,"dataset"): stack.append(d.dataset)
        elif hasattr(d,"ids"): out.extend(d.ids)
    return out
tr, va = ids_of(train_ds), ids_of(val_full)
tb, vb = {m1.base_id(i) for i in tr}, {m1.base_id(i) for i in va}
print(f"   train ids {len(tr):,} ({len(tb):,} base_ids) | val ids {len(va):,} ({len(vb):,} base_ids)")
print(f"   base_id overlap train∩val : {len(tb & vb)}   {'OK' if not (tb&vb) else '*** OVERLAP ***'}")
print(f"   exact id overlap          : {len(set(tr)&set(va))}   {'OK' if not (set(tr)&set(va)) else '*** OVERLAP ***'}")
al_t = {i.split('_')[0] for i in tr}; al_v = {i.split('_')[0] for i in va}
print(f"   allele overlap            : {len(al_t & al_v)} of {len(al_v)} val alleles "
      f"({'expected 0 for two_axis' if not (al_t&al_v) else 'SHARED — check the scheme'})")

print("="*78); print("3. UNTRAINED CONTROL — same metric, random weights")
for tag, model in (("TRAINED  ", net), ("UNTRAINED", build(None))):
    rp, rt, sp, st_ = [], [], [], []
    ld = m1.make_dataloader(val_full, 1, shuffle=False, num_workers=2)
    with torch.no_grad():
        for i, b in enumerate(ld):
            if i >= 100: break
            b = m1.move_batch(b, dev)
            o = model.predict(b)
            pl = m1.plddt_from_logits(o["plddt_logits"].float(), 50)[0].cpu().numpy()
            tp = b["teacher_plddt"][0].float().cpu().numpy()
            pep = m1.peptide_mask_from_batch(b["seq_mask"], b["segment_id"])[0].bool().cpu().numpy()
            rp.append(pl[pep]); rt.append(tp[pep]); sp.append(pl[pep].mean()); st_.append(tp[pep].mean())
    rp, rt = np.concatenate(rp), np.concatenate(rt)
    print(f"   {tag} per-residue r={pearsonr(rp,rt)[0]:+.3f} rho={spearmanr(rp,rt).statistic:+.3f} | "
          f"per-structure r={pearsonr(np.array(sp),np.array(st_))[0]:+.3f}")
print("   (a random model scoring well would mean the metric is not measuring learning)")
