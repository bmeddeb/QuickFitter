from __future__ import annotations
from lmfit.models import ComplexModel

from ._eval_funcs import hybrid_debye_lorentz_eval
from ._mixins import ScaledResidualMixin


class HybridDebyeLorentzModel(ScaledResidualMixin, ComplexModel):
    """Hybrid model combining Debye and Lorentz mechanisms."""

    def __init__(self, n_terms: int, prefix: str = "", **kws):
        self.n_terms = n_terms

        def _eval(f_ghz, eps_inf, **params):
            return hybrid_debye_lorentz_eval(f_ghz, n_terms, eps_inf, **params)

        super().__init__(_eval, independent_vars=["f_ghz"], prefix=prefix, **kws)

        self.set_param_hint("eps_inf", min=1.0)
        for i in range(1, n_terms + 1):
            self.set_param_hint(f"delta_eps_D{i}", min=0.0)
            self.set_param_hint(f"tau_D{i}", min=1e-15)
            self.set_param_hint(f"alpha{i}", min=0.0, max=1.0)
            self.set_param_hint(f"delta_eps_L{i}", min=0.0)
            self.set_param_hint(f"omega0{i}", min=1e7)
            self.set_param_hint(f"q{i}", min=0.0, max=1.0)
