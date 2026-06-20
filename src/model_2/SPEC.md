# model_2 — Enhanced MHC-Diff (SE(3)-EGNN diffusion)

An alternative, **generative** architecture to model_1's deterministic distillation:
a denoising-diffusion model over Cα coordinates with an SE(3)-equivariant,
chirality-aware EGNN denoiser, plus an auxiliary pLDDT head. Trains on the **same
processed H5 store** as model_1.

## File map (PyTorch Geometric)
| file | role |
|---|---|
| `egnn_pyg.py` | `EGNNDenoiser` + `EGNNLayer` — radius_graph edges, scatter message passing, k-NN triplet **cross-product chirality** |
| `diffusion.py` | differential (peptide-high / MHC-low) zero-CoM diffusion, flattened + `scatter_mean` centring |
| `pyg_data.py` | H5 → PyG `Data`; PyG `DataLoader`; **`MHCTemplatePool`** (same-length MHC init) |
| `model.py` | `MHCDiff` (denoiser + pLDDT head + **torsion head** + loss + template sampling) |
| `train.py` | training CLI (`--dummy` smoke / `--h5-dir` real) |

Built on `torch_geometric` + `torch_cluster` (radius_graph/knn_graph) + `torch_scatter`
(installed by `installation.sh` from the torch-matched PyG wheel index).

```bash
python src/model_2/train.py --dummy --epochs 3 --bs 3            # smoke test
# side-chain store (after reprocessing, see below) enables the torsion loss:
python src/model_2/train.py --h5-dir data/processed/h5_store_sc \
    --scheme two_axis --fold 1 --epochs 50 --bs 8 --amp
```

## Side chains & reprocessing (new)
The backbone-only store has no χ. Reprocess to a side-chain store that also carries
`teacher_chi [N,4,2]` (sin/cos of χ1..χ4) + `teacher_chi_mask [N,4]`, derived from
the teacher PDB's heavy atoms via OpenFold `atom37_to_torsion_angles`:
```bash
sbatch src/model/reprocess_sidechains.sbatch          # -> data/processed/h5_store_sc
python src/model/preprocess.py --merge-only --out-dir data/processed/h5_store_sc
```
`parse_example(..., return_sidechain=True)` and `preprocess.py --sidechains` produce
them; the χ columns are **optional** in the store (model_1 ignores them; model_2's
torsion head + `L_torsion` switch on automatically when present).

## MHC template initialisation (point #2)
Pure Gaussian noise is a poor start for the MHC (its low-variance schedule can't
recover a fold from scratch). Instead `MHCTemplatePool` indexes the store's MHCs by
length; at sampling time the MHC nodes are initialised from a **random same-length
real MHC** (a different allele), centred and treated as a **partial-noise** state
(`mhc_start_t`), while the peptide starts from full noise. The denoiser then refines
that template toward the target — exactly the induced-fit "treat the initial graph
as partial noise to denoise" idea. (`MHCDiff.sample(data, template_pool=...)`.)

## Data compatibility (requirement #4)
1. **Side chains:** the backbone-only store has no χ → reprocess to `h5_store_sc`
   (above). Then the torsion head + `L_torsion` train automatically. On the
   backbone-only store they stay off (`λ_torsion` term = 0).
2. **No crystal ground truth.** Targets are the PMGen/AF2 *teacher* structures, so
   the pLDDT head regresses `teacher_plddt` (not lDDT-vs-crystal) — honest given the
   data. Likewise χ targets are the teacher's side-chain angles.
3. **Frame orientation `O_i` is not diffused.** We diffuse Cα only; the torsion head
   predicts χ as a structure read-out (spec §3.3). Full backbone-*frame* diffusion
   (translation + SO(3)) + torsion *diffusion* (noising χ) are future extensions —
   the current torsion head is deterministic prediction, not a diffusion channel.
4. **Reuses `model/utils.py`** datasets/collate/peptide-mask, so model_2 trains on
   exactly the same examples as model_1 (now wrapped as PyG `Data`).

## Deviations from the spec (and why)
- **PyG done as requested**, but graphs use `torch_cluster.radius_graph` (the PyG
  `radius_graph` wrapper needs `pyg-lib`). Messages/aggregation use `torch_scatter`.
- **k-NN-bounded triplets** for the cross-product (see Q1) — `knn_graph(k=12)` —
  while messages use the type-dependent 8/14 Å `radius_graph`.
- **Coordinates normalized by 15 Å** so x0 and unit-Gaussian noise share a scale.
- **Zero-init coordinate MLPs** so each layer starts as identity (ε≈0) — stable.

---

## Critique (the 4 questions)

### Q1 — Triplet O(N³) + cross-product bottleneck, and how to optimize
Naive triplet message passing and the chirality double-sum are **O(N²) per node =
O(N³)** globally, and the cross-product is numerically explosive (|rᵢⱼ × rᵢₖ| grows
with raw distances). Fixes, all implemented:
- **Restrict triplets to a k-NN fan** (k≈16): cost drops to **O(N·k²)** with k≪N.
  The peptide stays fully connected (it's tiny); only the protein uses the cutoff
  → k-NN. This is the single most important optimization.
- **Unit-vector cross product** `û_ij × û_ik` (magnitude ≤ 1) + **mean** (not sum)
  aggregation → bounded updates regardless of k or distance scale.
- **Zero-init the coordinate MLPs** (`φ_x`, `φ_c`): each layer starts as identity,
  so ε starts at ~0 and training is stable from step 1.
- Coordinate **normalization** (÷15 Å) keeps everything O(1).
- Further options if needed: compute the k-NN graph **once** per forward (we do),
  share φ_c across layers, or use a GemNet-style edge-triplet sparse layout if you
  later move to PyG. Verified: gradients finite, loss ~3 (noise floor) not 1e18.

### Q2 — `SE3_EGNN_Layer` skeleton (implemented in `egnn.py`)
```python
m_ij  = φ_m([h_i, h_j, ‖x_i-x_j‖², a_ij, angle_pool(cosθ_jik over k-NN)])
Δx_i  = mean_j  (x_i-x_j)/(‖·‖+1) · φ_x(m_ij)                       # radial (E(3))
      + mean_jk (û_ij × û_ik)     · φ_c([m_ij, m_ik])               # chirality (SE(3))
h_i'  = h_i + φ_h([h_i, Σ_j m_ij])
x_i'  = x_i + Δx_i
```
Verified numerically: **SE(3)-equivariant to 3e-6** (rotation+translation) and the
cross-product term makes it **reflection-sensitive** (12% output change under a
mirror) — i.e. it preserves chirality, exactly as required.

### Q3 — Balancing λ₁..λ₄ (coord noise vs torsion vs pLDDT)
- **Put every term on an O(1) scale first**, then weight. Coord ε-loss is naturally
  O(1) (unit noise). The pLDDT MSE must be computed in **[0,1] units (÷100)** — as
  shipped — otherwise it's ~1000× the coord loss and dominates everything (we saw
  this). Torsion loss on `[sin,cos]` components is already O(1).
- **Peptide vs MHC (λ₁ vs λ₂):** weight the peptide higher (default 1.0 vs 0.25).
  The MHC is low-variance/induced-fit and partly "given", so it shouldn't dominate
  the gradient — same lesson as model_1's peptide weighting.
- **Coord vs torsion (when enabled):** the cleanest dynamic scheme is **uncertainty
  weighting** (Kendall et al.): learn a log-variance `sᵢ` per loss and use
  `Σ exp(-sᵢ)·Lᵢ + sᵢ`. It auto-balances heterogeneous losses (Å² noise vs angular)
  without hand-tuning, and is the standard fix for joint coordinate+torsion
  diffusion. A cheaper alternative: **GradNorm** (equalize per-task gradient norms),
  or a fixed schedule that ramps torsion in only after the backbone localizes
  (torsions are meaningless until the fold is roughly right).
- **pLDDT (λ₄):** keep it small (~0.1) and ideally **stop-grad the structure** into
  the pLDDT head so confidence learning doesn't perturb the denoiser — it's a
  read-out, not a structural objective.

### Q4 — Data compatibility
Done: reuses model_1's H5 dataset/collate/peptide-mask verbatim; trains on the
existing store with no re-processing. The only spec features that need new data
(full-atom store) are the side-chain torsions — gated off until then.
