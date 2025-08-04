"""Djordjevic–Sarkar dielectric dispersion model."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from .base_model import BaseModel


class DSarkarModel(BaseModel):
    """Implements the Djordjevic–Sarkar wideband dielectric model."""

    param_names = ["eps_inf", "delta_eps", "omega1", "omega2"]

    def epsilon(self, params: Dict[str, float], f_ghz: Iterable[float]) -> np.ndarray:  # type: ignore[override]
        f_ghz = np.asarray(f_ghz)
        omega = 2 * np.pi * f_ghz * 1e9
        eps_inf = params["eps_inf"]
        delta_eps = params["delta_eps"]
        omega1 = params["omega1"]
        omega2 = params["omega2"]

        with np.errstate(divide="ignore", invalid="ignore"):
            if omega2 <= 0 or omega1 <= 0 or omega2 <= omega1:
                return np.full_like(f_ghz, np.nan, dtype=complex)
            log_term = np.log((omega2**2 + omega**2) / (omega1**2 + omega**2))
            eps_prime = eps_inf + (delta_eps / (2 * np.log(omega2 / omega1))) * log_term
            atan_term = np.arctan(omega / omega1) - np.arctan(omega / omega2)
            eps_double_prime = -(delta_eps / np.log(omega2 / omega1)) * atan_term
        return eps_prime + 1j * eps_double_prime
