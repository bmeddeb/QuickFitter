"""Base classes for dielectric models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np
from lmfit import Minimizer, Parameters


class BaseModel(ABC):
    """Abstract base class for dielectric models.

    Sub-classes need to implement :meth:`epsilon` which returns the complex
    permittivity for a set of parameters and frequencies.  The base class
    provides a generic :meth:`residual` and :meth:`fit` implementation using
    :mod:`lmfit`.
    """

    @abstractmethod
    def epsilon(self, params: Dict[str, float], f_ghz: Iterable[float]) -> np.ndarray:
        """Return complex permittivity for the model.

        Parameters
        ----------
        params:
            Mapping of parameter names to values.
        f_ghz:
            Iterable of frequencies in GHz.
        """

    def residual(self, params: Parameters, f_ghz: np.ndarray, eps_data: np.ndarray) -> np.ndarray:
        """Residual used by :mod:`lmfit`.

        Real and imaginary parts are scaled independently by their dynamic
        range to avoid bias toward the component with the larger magnitude.
        """
        model_eps = self.epsilon(params.valuesdict(), f_ghz)
        scale_real = np.max(np.abs(np.real(eps_data))) or 1.0
        scale_imag = np.max(np.abs(np.imag(eps_data))) or 1.0
        real_res = (np.real(model_eps) - np.real(eps_data)) / scale_real
        imag_res = (np.imag(model_eps) - np.imag(eps_data)) / scale_imag
        return np.concatenate([real_res, imag_res])

    def fit(
        self,
        f_ghz: np.ndarray,
        eps_data: np.ndarray,
        init_params: Dict[str, float],
    ) -> 'lmfit.model.ModelResult':
        """Fit the model to complex permittivity data.

        Parameters
        ----------
        f_ghz:
            1‑D array of frequencies in GHz.
        eps_data:
            Complex permittivity values measured at ``f_ghz``.
        init_params:
            Dictionary with initial parameter guesses.
        """
        params = Parameters()
        for name, value in init_params.items():
            params.add(name, value=value, vary=True)

        minimizer = Minimizer(self.residual, params, f_ghz=np.asarray(f_ghz), eps_data=np.asarray(eps_data))
        return minimizer.minimize()
