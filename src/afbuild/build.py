"""One-shot: extract AF2 model_2_ptm weights into params/alphafold/ and verify.
Run once:  python src/afbuild/build.py
"""
import torch
from utils import REPO_ROOT, PARAMS_DIR, model_config, load_frozen_fold
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_

MODEL = "model_2_ptm"
NPZ   = REPO_ROOT / "params" / "alphafold" / f"params_{MODEL}.npz"


def extract():
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = model_config(MODEL)
    m = AlphaFold(cfg)                       # full model is just a vehicle
    import_jax_weights_(m, NPZ, version=MODEL)
    torch.save(m.structure_module.state_dict(), PARAMS_DIR / "sm_af2.pt")
    torch.save(m.aux_heads.plddt.state_dict(),  PARAMS_DIR / "plddt_af2.pt")
    torch.save(m.aux_heads.tm.state_dict(),     PARAMS_DIR / "pae_af2.pt")
    print(f"[build] saved sm/plddt/pae weights -> {PARAMS_DIR}")


def verify():
    ff = load_frozen_fold(MODEL, device="cpu")
    B, N = 1, 30
    s = torch.randn(B, N, 384, requires_grad=True)
    z = torch.randn(B, N, N, 128, requires_grad=True)
    aat  = torch.randint(0, 20, (B, N))
    mask = torch.ones(B, N)
    ca, plddt, pae = ff(s, z, aat, mask)
    assert ca.shape == (B, N, 3) and plddt.shape == (B, N, 50) and pae.shape == (B, N, N, 64)
    ca.sum().backward()
    assert s.grad is not None and z.grad is not None, "gradient not reaching encoder inputs"
    print(f"[build] verify OK  shapes={tuple(ca.shape)},{tuple(plddt.shape)},{tuple(pae.shape)}  grad_flow=True")


if __name__ == "__main__":
    extract()
    verify()