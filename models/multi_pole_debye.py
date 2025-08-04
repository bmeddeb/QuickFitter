from __future__ import annotations
from lmfit.models import ComplexModel

from ._eval_funcs import multi_pole_debye_eval
from ._mixins import ScaledResidualMixin


class MultiPoleDebyeModel(ScaledResidualMixin, ComplexModel):
    """Debye model with a configurable number of relaxation poles."""

    def __init__(self, n_poles: int, prefix: str = "", **kws):
        self.n_poles = n_poles

        def _eval(f_ghz, eps_inf, **params):
            return multi_pole_debye_eval(f_ghz, n_poles, eps_inf, **params)

        super().__init__(_eval, independent_vars=["f_ghz"], prefix=prefix, **kws)

        self.set_param_hint("eps_inf", min=1.0)
        for i in range(n_poles):
            self.set_param_hint(f"delta_eps_{i}", min=0.0)
            self.set_param_hint(f"log_tau_{i}", min=-15, max=5)
