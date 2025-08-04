"""Havriliak–Negami dielectric model."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .base_model import BaseModel


class HavriliakNegamiModel(BaseModel):
    """Implementation of the Havriliak–Negami relaxation model."""

    param_names = ["eps_inf", "delta_eps", "tau", "alpha", "beta"]

    def epsilon(self, params: Dict[str, float], f_ghz: Iterable[float]) -> np.ndarray:  # type: ignore[override]
        f_ghz = np.asarray(f_ghz)
        omega = 2 * np.pi * f_ghz * 1e9
        eps_inf = params["eps_inf"]
        delta_eps = params["delta_eps"]
        tau = params["tau"]
        alpha = params["alpha"]
        beta = params["beta"]

        with np.errstate(divide="ignore", invalid="ignore"):
            if tau <= 0 or alpha <= 0 or alpha > 1 or beta <= 0 or beta > 1:
                return np.full_like(f_ghz, np.nan, dtype=complex)
            jw_tau = 1j * omega * tau
            denominator = (1 + jw_tau ** alpha) ** beta
            eps_complex = eps_inf + delta_eps / denominator
        return eps_complex
