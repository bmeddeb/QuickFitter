"""Hybrid Debye–Lorentz model with independent Debye and Lorentz terms."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .base_model import BaseModel


class HybridDebyeLorentzModel(BaseModel):
    """Revised hybrid model combining Cole–Cole modified Debye and damped Lorentz terms."""

    def __init__(self, n_terms: int):
        self.n_terms = n_terms

    def epsilon(self, params: Dict[str, float], f_ghz: Iterable[float]) -> np.ndarray:  # type: ignore[override]
        f_ghz = np.asarray(f_ghz)
        omega = 2 * np.pi * f_ghz * 1e9
        eps_inf = float(params.get("eps_inf", 1.0))
        complex_eps = np.ones_like(omega, dtype=complex) * eps_inf

        for i in range(self.n_terms):
            delta_D = float(params.get(f"delta_eps_D{i+1}", 0.0))
            tau_D = float(params.get(f"tau_D{i+1}", 1e-9))
            alpha = float(params.get(f"alpha{i+1}", 1.0))
            delta_L = float(params.get(f"delta_eps_L{i+1}", 0.0))
            omega0 = float(params.get(f"omega0{i+1}", 1e10))
            q = float(params.get(f"q{i+1}", 0.05))
            gamma = q * omega0

            debye = delta_D / (1 + (1j * omega * tau_D) ** alpha)
            lorentz = delta_L * omega0 ** 2 / (omega0 ** 2 - omega ** 2 - 1j * 2 * gamma * omega)
            complex_eps += debye + lorentz

        return complex_eps
