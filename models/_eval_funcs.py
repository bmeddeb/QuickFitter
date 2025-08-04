from __future__ import annotations
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

_GHZ_TO_RAD_S = 2 * np.pi * 1e9


def hn_eval(
    f_ghz: FloatArray,
    eps_inf: float,
    delta_eps: float,
    tau: float,
    alpha: float,
    beta: float,
) -> ComplexArray:
    """Havriliak–Negami permittivity in SI units internally."""
    omega = _GHZ_TO_RAD_S * f_ghz
    jw_tau = 1j * omega * tau
    return eps_inf + delta_eps / (1 + jw_tau ** alpha) ** beta


def dsarkar_eval(
    f_ghz: FloatArray,
    eps_inf: float,
    delta_eps: float,
    omega1: float,
    omega2: float,
) -> ComplexArray:
    """Djordjevic–Sarkar permittivity evaluated over ``f_ghz``."""
    omega = _GHZ_TO_RAD_S * f_ghz
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log((omega2**2 + omega**2) / (omega1**2 + omega**2))
        eps_prime = eps_inf + (delta_eps / (2 * np.log(omega2 / omega1))) * log_term
        atan_term = np.arctan(omega / omega1) - np.arctan(omega / omega2)
        eps_double_prime = -(delta_eps / np.log(omega2 / omega1)) * atan_term
    return eps_prime + 1j * eps_double_prime


def multi_pole_debye_eval(
    f_ghz: FloatArray,
    n_poles: int,
    eps_inf: float,
    **params: float,
) -> ComplexArray:
    """Multi-pole Debye model evaluation.

    Parameters are expected as ``delta_eps_i`` and either ``log_tau_i`` or
    ``tau_i`` for ``i`` in ``range(n_poles)``.
    """
    omega = _GHZ_TO_RAD_S * f_ghz
    perm = np.full_like(omega, eps_inf, dtype=np.complex128)
    for i in range(n_poles):
        delta = params.get(f"delta_eps_{i}", 0.0)
        if f"log_tau_{i}" in params:
            tau = 10 ** params[f"log_tau_{i}"]
        else:
            tau = params.get(f"tau_{i}", 1e-12)
        perm += delta / (1 + 1j * omega * tau)
    return perm


def hybrid_debye_lorentz_eval(
    f_ghz: FloatArray,
    n_terms: int,
    eps_inf: float,
    **params: float,
) -> ComplexArray:
    """Hybrid Debye–Lorentz model evaluation."""
    omega = _GHZ_TO_RAD_S * f_ghz
    eps = np.full_like(omega, eps_inf, dtype=np.complex128)
    for i in range(n_terms):
        j = i + 1
        delta_D = params.get(f"delta_eps_D{j}", 0.0)
        tau_D = params.get(f"tau_D{j}", 1e-12)
        alpha = params.get(f"alpha{j}", 1.0)
        delta_L = params.get(f"delta_eps_L{j}", 0.0)
        omega0 = params.get(f"omega0{j}", 1.0)
        q = params.get(f"q{j}", 0.0)
        gamma = q * omega0
        debye = delta_D / (1 + (1j * omega * tau_D) ** alpha)
        lorentz = delta_L * omega0**2 / (
            omega0**2 - omega**2 - 1j * 2 * gamma * omega
        )
        eps += debye + lorentz
    return eps
