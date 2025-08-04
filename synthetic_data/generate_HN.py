import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------- Havriliak–Negami function --------------------
def eps_hn(f_hz, eps_inf, branches):
    """
    Compute complex permittivity for a multi‑branch HN model.

    Parameters
    ----------
    f_hz : ndarray
        Frequencies [Hz].
    eps_inf : float
        High‑frequency permittivity.
    branches : list[dict]
        Each dict has keys
           'd_eps' : Δε
           'tau'   : relaxation time [s]
           'alpha' : 0–1 (Cole–Cole)
           'beta'  : 0–1 (Cole–Davidson)
    """
    w = 2*np.pi*f_hz
    eps = eps_inf + 0j
    for b in branches:
        d_eps, tau, a, bta = (
            b["d_eps"], b["tau"], b["alpha"], b["beta"])
        eps += d_eps / (1 + (1j*w*tau)**a) ** bta
    return eps

# --------------------------- frequency grid -----------------------------
freq_hz = np.logspace(3, 12.3, 2300)   # 1 kHz – 2 THz

# --------------------------- scenario table -----------------------------
scenarios = {
    # 1) Pure Debye – reference
    "S1_Debye" : dict(
        eps_inf = 2.1,
        branches = [
            dict(d_eps=4.5, tau=18e-9, alpha=1.0, beta=1.0)
        ]
    ),
    # 2) Cole–Cole (symmetric broadening)
    "S2_ColeCole" : dict(
        eps_inf = 3.5,
        branches = [
            dict(d_eps=7.0, tau=4.5e-6, alpha=0.72, beta=1.0)
        ]
    ),
    # 3) Cole–Davidson (asymmetric broadening)
    "S3_ColeDavidson" : dict(
        eps_inf = 2.8,
        branches = [
            dict(d_eps=5.0, tau=2.0e-4, alpha=1.0, beta=0.55)
        ]
    ),
    # 4) Full Havriliak–Negami (both α and β < 1)
    "S4_HN" : dict(
        eps_inf = 2.2,
        branches = [
            dict(d_eps=3.8, tau=6.0e-7, alpha=0.63, beta=0.42)
        ]
    ),
    # 5) Two overlapping relaxations (polymer glassy + secondary β‑process)
    "S5_Overlap" : dict(
        eps_inf = 2.0,
        branches = [
            dict(d_eps=6.0, tau=1.2e-2, alpha=0.85, beta=0.60),
            dict(d_eps=1.8, tau=4.0e-5, alpha=0.60, beta=0.35)
        ]
    ),
    # 6) Microwave Debye plus ultra‑fast (sub‑ps) tail
    "S6_MW_plus_THz" : dict(
        eps_inf = 4.4,
        branches = [
            dict(d_eps=50.0,  tau=8.3e-12, alpha=1.0, beta=1.0),   # water‑like MW
            dict(d_eps=1.2,   tau=0.25e-12, alpha=0.90, beta=0.90) # few‑THz tail
        ]
    ),
}

# --------------------------- noise level --------------------------------
NOISE_DK = 0.03       # absolute σ for ε′
NOISE_DF = 5e-4       # absolute σ for tanδ

# ------------------------ generate and export ---------------------------
out_dir = Path("synthetic_HN_data")
out_dir.mkdir(exist_ok=True)

for tag, cfg in scenarios.items():
    eps = eps_hn(freq_hz, cfg["eps_inf"], cfg["branches"])
    dk = np.real(eps)
    df = -np.imag(eps) / dk

    # add optional noise
    rng = np.random.default_rng(seed=hash(tag) & 0xffffffff)
    dk += rng.normal(0, NOISE_DK, size=dk.size)
    df += rng.normal(0, NOISE_DF, size=df.size)
    df[df < 0] = 0        # keep physical

    pd.DataFrame({
        "Frequency_GHz": freq_hz / 1e9,
        "Dk": dk,
        "Df": df
    }).to_csv(out_dir / f"{tag}.csv", index=False)

    print(f"✓  {tag}  →  {out_dir / (tag+'.csv')}")
