"""Multi‑pole Debye relaxation model."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .base_model import BaseModel


class MultiPoleDebyeModel(BaseModel):
    """Debye model with an arbitrary number of relaxation poles."""

    def __init__(self, n_poles: int):
        self.n_poles = n_poles

    def epsilon(self, params: Dict[str, float], f_ghz: Iterable[float]) -> np.ndarray:  # type: ignore[override]
        f_ghz = np.asarray(f_ghz)
        omega = 2 * np.pi * f_ghz * 1e9
        eps_inf = float(params.get("eps_inf", 1.0))
        complex_eps = np.ones_like(omega, dtype=complex) * eps_inf

        for i in range(self.n_poles):
            delta_eps = float(params.get(f"delta_eps_{i}", 0.0))
            # Support either log_tau_i or tau_i
            if f"log_tau_{i}" in params:
                tau = 10 ** float(params[f"log_tau_{i}"])
            else:
                tau = float(params.get(f"tau_{i}", 1e-12))
            complex_eps += delta_eps / (1 + 1j * omega * tau)

        return complex_eps
