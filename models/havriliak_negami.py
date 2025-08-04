from __future__ import annotations
import numpy as np
import numpy.typing as npt
from lmfit.models import ComplexModel

from ._eval_funcs import hn_eval
from ._mixins import ScaledResidualMixin

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


class HavriliakNegamiModel(ScaledResidualMixin, ComplexModel):
    """lmfit ComplexModel wrapper for the Havriliak–Negami dispersion."""

    def __init__(self, prefix: str = "", **kws):
        super().__init__(hn_eval, independent_vars=["f_ghz"], prefix=prefix, **kws)
        self.set_param_hint("eps_inf", value=1.0, min=1.0)
        self.set_param_hint("delta_eps", value=1.0, min=0.0)
        self.set_param_hint("tau", value=1e-9, min=1e-15)
        self.set_param_hint("alpha", value=0.8, min=0.0, max=1.0)
        self.set_param_hint("beta", value=0.8, min=0.0, max=1.0)

    def guess(
        self, data: ComplexArray, f_ghz: FloatArray, **overrides: float
    ):
        peak_idx = np.argmax(np.imag(data))
        tau_guess = 1 / (2 * np.pi * f_ghz[peak_idx] * 1e9)
        params = self.make_params(
            eps_inf=np.real(data[-1]),
            delta_eps=np.real(data[0]) - np.real(data[-1]),
            tau=tau_guess,
            alpha=0.8,
            beta=0.8,
        )
        params.update(overrides)
        return params
