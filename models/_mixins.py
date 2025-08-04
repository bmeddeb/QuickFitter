from __future__ import annotations
import numpy as np


class ScaledResidualMixin:
    """Normalise real and imag residuals by their dynamic range."""

    def _residual(self, params, data, f_ghz=None, **kws):  # lmfit hook
        model = self.eval(params, f_ghz=f_ghz, **kws)
        scale_r = np.ptp(np.real(data)) or 1.0
        scale_i = np.ptp(np.imag(data)) or 1.0
        res = (np.real(model) - np.real(data)) / scale_r
        resi = (np.imag(model) - np.imag(data)) / scale_i
        return np.concatenate((res, resi)).ravel()
