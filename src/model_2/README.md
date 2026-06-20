# model_2 — MHC-Diff: an SE(3)-equivariant diffusion generator for pMHC

An **alternative** to model_1. Where model_1 deterministically distills AF2 in one
forward pass, model_2 is a **generative denoising-diffusion model** over the
complex's Cα coordinates, denoised by an **SE(3)-equivariant Graph Neural Network**
(EGNN) in PyTorch Geometric. It can sample structures and gives a confidence
(pLDDT) and, on the side-chain store, side-chain torsions.

This file explains *exactly* what the model is, what happens when it runs, and what
it optimizes. (`SPEC.md` covers the design critique and deviations from the original
spec; this README is the "how it works".)

---

## 1. What the model sees (inputs)

One training example = one pMHC complex, taken from the **same processed H5 store as
model_1** (targets are the PMGen/AF2 *teacher* structures). Per residue we have:

| field | meaning |
|---|---|
| `teacher_ca` `[N,3]` | Cα coordinates (the diffusion variable / target) |
| `aatype` `[N]` | amino-acid identity |
| `segment_id` `[N]` | 0 = MHC, 1 = peptide (peptide = highest segment) |
| `residue_index` `[N]` | residue numbering (carries the ~200 MHC↔peptide gap) |
| `teacher_plddt` `[N]` | teacher per-residue confidence (pLDDT regression target) |
| `teacher_chi` `[N,4,2]` *(side-chain store only)* | χ1–χ4 as (sin, cos) — torsion target |

Residue order is **MHC first, then peptide**. Each example becomes a
`torch_geometric.data.Data`; PyG batches variable-N graphs with no padding.

---

## 2. The geometric graph

Each residue is **one node** placed at its Cα. Coordinates are normalized by **15 Å**
so they're ~unit scale (matching the unit-Gaussian diffusion noise), and every graph
is **zero-centre-of-mass** (per-graph, via `scatter_mean`).

**Node features** `h⁰`: aatype one-hot + segment one-hot + sinusoidal positional
encoding of `residue_index` + the diffusion time `t/T`.

**Edges** (directed, deduped — built in `build_message_edges`):
1. **protein–protein** within **8 Å**, **protein–peptide** within **14 Å** (`radius_graph`);
2. **peptide = complete subgraph** — *all* intra-peptide pairs, any distance, so the
   peptide is **always internally connected** even when it's pure noise;
3. **peptide ↔ MHC k-NN** — each peptide residue is linked to its **6 nearest MHC
   residues** regardless of distance, so the peptide always "feels" the groove.

(2) and (3) exist specifically so a noisy, scattered peptide is never disconnected.
**Edge features**: edge type one-hot (pep-pep / pro-pro / mixed), same-chain flag,
and clamped sequence separation.

> The graph is rebuilt **once per denoiser call** (i.e. once per diffusion step) from
> the current coordinates — not per EGNN layer. So as the peptide takes shape over the
> reverse process, its edges are re-derived at every step.

---

## 3. The forward diffusion (what corrupts the data)

Training corrupts the clean Cα cloud `x₀` to a noised `xₜ` at a random timestep `t`,
with **differential, per-segment noise** (`DifferentialDiffusion`):

- **Peptide** — full-variance cosine schedule: at high `t` it becomes ~`N(0, I)`
  (must be generated from scratch).
- **MHC** — low-variance schedule (β × 0.1): only mildly perturbed, so its global
  fold is preserved (models induced-fit flexibility).

Formally, per node `xₜ = √ᾱ · x₀ + √(1-ᾱ) · ε`, where `ᾱ` is taken from the peptide or
MHC schedule depending on the node's segment, `ε ~ N(0, I)`, and both `x₀` and `ε` are
projected to the **zero-CoM subspace** (so `xₜ` is too). The network's job is to
predict `ε`.

---

## 4. The denoiser network (EGNN)

`EGNNDenoiser`: embed `h⁰`, run **L equivariant layers**, and output the **predicted
noise** as the net coordinate displacement `ε̂ = x_L − x_in` (an equivariant vector).

Each `EGNNLayer` does:
```
m_ij = φ_m([h_i, h_j, ‖x_i−x_j‖², edge_attr])                      # invariant message
Δx_i = mean_j  (x_i−x_j)/(‖·‖+1) · φ_x(m_ij)                       # radial update  (E(3))
     + mean_jk (û_ij × û_ik)     · φ_c([m_ij, m_ik])               # chirality update (SE(3))
h_i' = h_i + φ_h([h_i, Σ_j m_ij])                                  # node update
x_i' = x_i + Δx_i
```
- Messages use only **invariant scalars** → the update is built from **equivariant
  vectors** (coordinate differences and cross-products) scaled by invariants. Result:
  **exactly SE(3)-equivariant** (translation + rotation), verified to ~1e-6.
- The **cross-product** term is odd under reflection, so it **breaks E(3)→SE(3)** —
  the network is **chirality-aware** (won't predict mirror-image structures).
- The chirality term sums over **triplets** (pairs of a node's edges). To keep it from
  being O(N³), triplets come from a bounded **k-NN graph** (k≈12) → O(N·k²), built
  with a fully-vectorized pairing (no Python loops).
- The coordinate MLPs (`φ_x`, `φ_c`) are **zero-initialized**, so the model starts as
  the identity (ε̂ ≈ 0) and training is stable from step 1.

### One training step, end to end (`MHCDiff.compute_loss`)
1. `x₀ = center(teacher_ca / 15)`, identify peptide vs MHC nodes.
2. Sample `t` per graph; `xₜ, ε = diffusion.q_sample(x₀, t, …)`.
3. Build node features `h⁰` (with `t/T`); run the denoiser on `xₜ` → `ε̂`.
4. Coordinate loss = MSE(`ε̂`, `ε`), split into peptide and MHC.
5. A **separate 2-layer EGNN** runs on the **clean** `x₀` (no time) to produce
   embeddings for the auxiliary heads (below).

---

## 5. Auxiliary heads

- **pLDDT head**: MLP on the clean-structure embedding → per-residue confidence in
  `[0,100]`, trained to regress `teacher_plddt`.
- **Torsion head** *(only when `teacher_chi` is present)*: MLP → χ1–χ4 as (sin, cos),
  normalized to the unit circle, trained against the teacher torsions. This is a
  *deterministic* side-chain predictor (spec §3.3), not yet a torsion diffusion channel.

---

## 6. Loss objectives

```
L = λ1·L_pep_coord + λ2·L_mhc_coord + λ3·L_torsion + λ4·L_pLDDT
```
| term | what | form | default λ |
|---|---|---|---|
| `L_pep_coord` | peptide structure (the hard part) | MSE(ε̂, ε) over peptide nodes | **1.0** |
| `L_mhc_coord` | MHC structure (induced fit) | MSE(ε̂, ε) over MHC nodes | **0.25** |
| `L_torsion` | side-chain χ | (sin,cos) MSE, masked by which χ exist | **0.5** *(0 if no side chains)* |
| `L_pLDDT` | confidence | MSE in [0,1] (÷100) vs teacher pLDDT | **0.1** |

Balancing notes: every term is kept **O(1)** (coord losses are unit-scale ε; pLDDT is
divided by 100; torsion is on the unit circle). The peptide is up-weighted vs the MHC
on purpose — the MHC is low-variance/partly-given and shouldn't dominate the gradient.

---

## 7. Inference (sampling)

`MHCDiff.sample(data, template_pool=…, mhc_start_t=…)` runs the reverse diffusion:
- **Peptide** starts from Gaussian noise.
- **MHC** starts from a **same-length real MHC template** (a different allele, via
  `MHCTemplatePool`) treated as a **partial-noise** state — a structural prior the
  denoiser refines toward the target sequence (induced fit), rather than hallucinating
  a fold from pure noise.
- Each reverse step calls the denoiser (which **re-derives the graph** from the current
  coordinates), so the peptide's connectivity tracks its emerging shape.

**Speed.** One denoiser call is **~4–6 ms (batch 1)** at pMHC sizes — there is no
frozen AF2 stack, and the graph additions cost ~nothing. Total inference time =
(#reverse steps) × (per-step). Full DDPM (`T=200`) ≈ 1 s; for a **50–100 ms** budget
use **few-step sampling** (~10–20 steps), which is a sampler choice independent of the
network.

---

## 8. Files & commands

| file | role |
|---|---|
| `egnn_pyg.py` | edges (radius + peptide-complete + pep↔MHC kNN), EGNN layer, denoiser |
| `diffusion.py` | differential zero-CoM diffusion schedules + q_sample + reverse |
| `pyg_data.py` | H5 → PyG `Data`, DataLoader, `MHCTemplatePool` |
| `model.py` | `MHCDiff` (denoiser + pLDDT/torsion heads + loss + sampling) |
| `train.py` | training CLI |

```bash
# smoke test (15 dummy examples)
python src/model_2/train.py --dummy --epochs 3 --bs 3

# real run on the side-chain store (enables the torsion loss)
python src/model_2/train.py --h5-dir data/processed/h5_store_sc \
    --scheme two_axis --fold 1 --epochs 50 --bs 8 --amp
```
Key flags: `--layers`, `--k` (triplet degree), `--timesteps`, `--mhc-scale`
(MHC vs peptide noise), `--lambdas PEP MHC TORSION PLDDT`, `--no-cross` (drop
chirality → E(3) only), `--amp`.

Build the side-chain store first with `src/model/reprocess_sidechains.sbatch`
(then `preprocess.py --merge-only`); see `SPEC.md`.
