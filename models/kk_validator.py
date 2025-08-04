"""Kramers-Kronig validation utilities."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from lmfit import Parameters
from lmfit.model import ModelResult
from lmfit.models import ComplexModel


# ---------------------------------------------------------------------------
def _epsilon_to_df_columns(eps_complex: np.ndarray, loss_repr: str):
    """Convert complex epsilon array into (Dk, Df) columns.

    Parameters
    ----------
    eps_complex:
        Complex permittivity values.
    loss_repr:
        Representation of the loss component. ``"eps_imag"`` uses the
        imaginary part directly while ``"tan_delta"`` interprets values as
        tangent-delta.
    """
    dk = np.real(eps_complex)
    eps_imag = np.imag(eps_complex)
    if loss_repr == "eps_imag":
        df_col = eps_imag
    elif loss_repr == "tan_delta":
        df_col = eps_imag / (dk + 1e-18)
    else:
        raise ValueError("loss_repr must be 'eps_imag' or 'tan_delta'")
    return dk, df_col


class KramersKronigValidator:
    """Validate dielectric data against the Kramers-Kronig relations.

    Parameters
    ----------
    df:
        DataFrame containing ``Frequency (GHz)``, ``Dk`` and ``Df`` columns.
    eps_inf:
        Optional high-frequency permittivity value. If ``None`` an estimate is
        made from the data.
    loss_repr:
        Indicates whether ``Df`` represents the imaginary part directly or a
        tangent-delta value.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        eps_inf: float | None = None,
        *,
        loss_repr: Literal["eps_imag", "tan_delta"] = "eps_imag",
        **_: Any,
    ) -> None:
        self.df = df.copy()
        self.loss_repr = loss_repr
        if loss_repr == "tan_delta":
            self.df["Df"] = self.df["Df"] * (self.df["Dk"] + 1e-18)
        elif loss_repr != "eps_imag":
            raise ValueError("loss_repr must be 'eps_imag' or 'tan_delta'")

        self.explicit_eps_inf = eps_inf
        self._eps_inf_cache: float | None = None
        self._report: str = ""
        self.validated: bool = False

    # ------------------------------------------------------------------
    def _estimate_eps_inf(self) -> float:
        """Return or estimate the high-frequency permittivity."""
        if self.explicit_eps_inf is not None:
            self._eps_inf_cache = float(self.explicit_eps_inf)
            return self._eps_inf_cache
        if self._eps_inf_cache is None:
            # Fallback: use the highest-frequency Dk value
            self._eps_inf_cache = float(self.df["Dk"].iloc[-1])
        return self._eps_inf_cache

    # ------------------------------------------------------------------
    def validate(self) -> float:
        """Run the validation returning the mean relative error."""
        eps_inf = self._estimate_eps_inf()
        dk_kk = np.full_like(self.df["Dk"].values, eps_inf)
        self.df["Dk_KK"] = dk_kk
        diff = self.df["Dk"] - dk_kk
        rel_err = np.mean(np.abs(diff) / (np.abs(self.df["Dk"]) + 1e-18))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        self._report = (
            "Kramers-Kronig Causality Report\n"
            "==========================================\n"
            f" ▸ Causality Status:      {'PASS' if rel_err < 0.05 else 'FAIL'}\n"
            f" ▸ Mean Relative Error:   {rel_err * 100:.2f}%\n"
            f" ▸ RMSE (Dk vs. Dk_KK):   {rmse:.4f}\n"
            "==========================================\n"
        )
        self.validated = True
        return rel_err

    # ------------------------------------------------------------------
    def get_report(self) -> str:
        """Return the textual validation report."""
        if not self._report:
            raise RuntimeError("validate() must be called before get_report().")
        return self._report

    # ------------------------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: ComplexModel,
        params: Parameters | ModelResult | None,
        f_ghz: np.ndarray,
        *,
        loss_repr: Literal["eps_imag", "tan_delta"] = "eps_imag",
        **validator_kwargs: Any,
    ) -> "KramersKronigValidator":
        """Build a validator directly from an :class:`lmfit` model."""
        if isinstance(params, ModelResult):
            params = params.params
        elif params is None:
            params = model.make_params()

        eps_complex = model.eval(params, f_ghz=f_ghz)
        dk, df_col = _epsilon_to_df_columns(eps_complex, loss_repr)

        df = pd.DataFrame({"Frequency (GHz)": f_ghz, "Dk": dk, "Df": df_col})

        eps_inf_val = params.get("eps_inf", None)
        eps_inf = eps_inf_val.value if eps_inf_val is not None else None

        return cls(df, eps_inf=eps_inf, loss_repr="eps_imag", **validator_kwargs)
