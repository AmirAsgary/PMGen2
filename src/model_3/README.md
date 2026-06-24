# model_3 — distill a truncated, mostly-frozen slice of the real AF2 Evoformer

Instead of model_1's 1.8 M from-scratch encoder, model_3 reuses the **real pretrained
AF2 Evoformer**, keeps only its **first K blocks** (`--evo-layers`, default 3), fine-tunes
just the **last block's last `--trainable` %** (default 10 %), and feeds the **frozen**
AF2 structure module + pLDDT/PAE heads — distilling against the teacher structures
(FAPE + pLDDT/PAE CE, reused from model_1). The representation the SM needs provably
exists (AF2 produces it); we only adapt a thin slice.

## Inputs (single-sequence, no MSA, no templates)
Same H5 store and per-residue fields as model_1/2. Built into AF2 features by
[`featurize.py`](featurize.py):
- `target_feat[B,N,22] = [has_break, one_hot(aatype,21)]`
- `msa_feat[B,1,N,49]  = [one_hot(aatype,23), 0, 0, one_hot(aatype,23), 0]` (depth-1 query)
- Multimer "gap trick" is **already in the data** — `residue_index` carries PMGen's
  ~200 MHC→peptide jump (AF2 relpos clips at ±32 → reads as separate chains).
  `anchor_canonical_resindex` renumbers the peptide from its anchors (register), preserving
  that gap. No new params.

## Files
- `featurize.py` — our batch → AF2 input dict (verified against `data_transforms.make_msa_feat`).
- `model.py` — `EvoDistillModel`: builds full `AlphaFold(model_config("model_2_ptm"))`,
  `import_jax_weights_` from `params/alphafold/params_model_2_ptm.npz`, keeps
  `input_embedder` + `evoformer.blocks[:K]` (+ frozen SM/heads), unfreezes the last
  `trainable` % of `blocks[K-1]`. Same `forward(batch) -> (ca, plddt, pae[, frames])`
  contract as `DistillModel`, so it reuses model_1's loss + loop.
- `train.py` — DDP training; checkpoints **only the trainable params** (the rest reload
  from the npz at init).
- `train_af3.sbatch` — torchrun on all visible GPUs, bs 1–2 (real Evoformer is heavy).

## Run

```bash
PY=$(conda info --base)/envs/pmgen2/bin/python

# 0) UNIT CHECK (do this first): features wire up + untrained forward is sane.
#    A garbage MHC RMSD here => the AF2 feature layout is wrong.
$PY - <<'EOF'
import sys; sys.path.insert(0, "src/model_3")
import torch, model as M3
U = M3.m1
net = M3.EvoDistillModel(evo_layers=3, trainable=10.0,
                         device="cuda" if torch.cuda.is_available() else "cpu")
net.eval()
ds = U.build_h5_dataset("data/processed/h5_store", "two_axis", 1, "val")
loader = U.make_dataloader(ds, 1, shuffle=False, num_workers=0)
batch = U.move_batch(next(iter(loader)), next(net.parameters()).device)
with torch.no_grad():
    ca, plddt, pae, frames = net(batch, return_frames=True)
print("shapes:", tuple(ca.shape), tuple(plddt.shape), tuple(pae.shape),
      "finite:", bool(torch.isfinite(ca).all()))
sm = batch["seq_mask"]; pep = U.peptide_mask_from_batch(sm, batch["segment_id"])
m = sm[0].bool(); pm = pep[0].bool() & m; am = (~pep[0].bool()) & m
print("UNTRAINED  MHC-RMSD %.2f  pep-RMSD %.2f Å (MHC should be reasonable)" % (
    U._superpose_rmsd_on(ca[0], batch["teacher_ca"][0], am, am),
    U._superpose_rmsd_on(ca[0], batch["teacher_ca"][0], am, pm)))
EOF

# 1) SMOKE (1 GPU, dummy set): loss decreases, checkpoint + resume work.
$PY src/model_3/train.py --dummy --epochs 2 --bs 1 --evo-layers 3 --trainable 10

# 2) FULL (DDP):
sbatch src/model_3/train_af3.sbatch        # checkpoints_model3/af3_two_axis_fold1_K3/

# 3) COMPARE to model_1/2:
$PY src/post_structure_prediction_processing/eval_stratified.py --model 3 \
    --ckpt checkpoints_model3/af3_two_axis_fold1_K3/last.pt \
    --h5-dir data/processed/h5_store --scheme two_axis --fold 1 \
    --out-dir outputs/eval_stratified/model3
```

## Knobs / notes
- `--evo-layers` (K) and `--trainable` (%) are the two levers: if val peptide-RMSD
  won't move, raise `--trainable` (e.g. 25–50) or K. The frozen SM caps quality at the
  teacher pLDDT≈51 ceiling, same as model_1/2.
- **Heavy**: a real Evoformer block (triangle attention/multiplication) is far heavier
  than model_1's encoder; gradient checkpointing is on, bs is 1–2. The full AF2 model is
  built per rank at init (→ `--mem=240G`).
- Single-seq + truncated + no-MSA is out-of-distribution for the pretrained weights; the
  MHC (conserved) should still fold, the peptide is the open question that fine-tuning
  the trainable slice is meant to answer.
