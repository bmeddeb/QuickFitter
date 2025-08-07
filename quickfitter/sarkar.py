"""Core algorithms for Djordjevic-Sarkar model fitting."""

from __future__ import annotations

import numpy as np


def estimate_initial_parameters(f_ghz: np.ndarray, complex_epsilon: np.ndarray) -> dict:
    """Estimate initial model parameters from frequency and permittivity data."""
    dk = np.real(complex_epsilon)
    eps_inf_init = dk[-1] if len(dk) > 0 else 1.0
    eps_s_init = dk[0] if len(dk) > 0 else 2.0
    delta_eps_init = eps_s_init - eps_inf_init
    omega1_init = 2 * np.pi * (f_ghz[0] if len(f_ghz) > 0 else 1.0) * 1e9
    omega2_init = 2 * np.pi * (f_ghz[-1] if len(f_ghz) > 0 else 100.0) * 1e9
    return {
        "eps_inf": eps_inf_init,
        "delta_eps": delta_eps_init,
        "omega1": omega1_init,
        "omega2": omega2_init,
    }


def calculate_model(
    eps_inf: float,
    delta_eps: float,
    omega1: float,
    omega2: float,
    f_ghz: np.ndarray,
) -> np.ndarray:
    """Compute complex permittivity for the Djordjevic-Sarkar model."""
    omega = 2 * np.pi * f_ghz * 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        if omega2 <= 0 or omega1 <= 0 or omega2 <= omega1:
            eps_prime = np.full_like(f_ghz, np.nan)
            eps_double_prime = np.full_like(f_ghz, np.nan)
        else:
            log_term = np.log((omega2**2 + omega**2) / (omega1**2 + omega**2))
            eps_prime = eps_inf + (delta_eps / (2 * np.log(omega2 / omega1))) * log_term
            atan_term = np.arctan(omega / omega1) - np.arctan(omega / omega2)
            eps_double_prime = -(delta_eps / np.log(omega2 / omega1)) * atan_term
    return eps_prime + 1j * eps_double_prime


def residual(params: dict, f_ghz: np.ndarray, complex_epsilon_data: np.ndarray) -> np.ndarray:
    """Return scaled residuals between model and measured data."""
    model_epsilon = calculate_model(
        params["eps_inf"],
        params["delta_eps"],
        params["omega1"],
        params["omega2"],
        f_ghz,
    )
    scale_real = np.max(np.abs(np.real(complex_epsilon_data))) or 1.0
    scale_imag = np.max(np.abs(np.imag(complex_epsilon_data))) or 1.0
    real_residual = (np.real(model_epsilon) - np.real(complex_epsilon_data)) / scale_real
    imag_residual = (np.imag(model_epsilon) - np.imag(complex_epsilon_data)) / scale_imag
    return np.concatenate([real_residual, imag_residual])
