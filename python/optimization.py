"""
optimization.py - Optimization routines for dielectric fitting
"""
import numpy as np
from typing import Tuple, Any
from lmfit import Parameters, Minimizer

from data_structures import FitData, FitParameters
from models import DjordjevicSarkarModel


class Optimizer:
    """Handles the optimization process"""

    def __init__(self, fit_data: FitData):
        self.fit_data = fit_data

    def create_parameters(self, initial: FitParameters) -> Parameters:
        """Create lmfit Parameters object with bounds"""
        params = Parameters()
        dk_data = np.real(self.fit_data.complex_eps)

        params.add('eps_inf', value=initial.eps_inf, min=1.0, max=dk_data.max())
        params.add('delta_eps', value=initial.delta_eps, min=0, max=10)

        # Allow wider frequency range for omega1 and omega2
        omega1_min = 2 * np.pi * 0.01e9   # 0.01 GHz minimum
        omega2_max = 2 * np.pi * 10000e9  # 10 THz maximum

        params.add('omega1', value=initial.omega1, min=omega1_min, max=omega2_max)
        params.add('omega2', value=initial.omega2, min=omega1_min, max=omega2_max)

        return params

    def jacobian_wrapper(self, params: Parameters, f_ghz: np.ndarray, complex_eps: np.ndarray) -> np.ndarray:
        """Wrapper to use complex-step Jacobian with lmfit"""
        return DjordjevicSarkarModel.complex_step_jacobian(params.valuesdict(), f_ghz)

    def optimize(self, method: str = 'leastsq') -> Tuple[FitParameters, Any]:
        """Run optimization and return fitted parameters"""
        # Get initial guess
        initial = DjordjevicSarkarModel.estimate_initial_parameters(self.fit_data)

        # Create parameters
        params = self.create_parameters(initial)

        # Map method names from HTML to lmfit
        method_map = {
            'least_squares': 'leastsq',
            'leastsq': 'leastsq',
            'nelder': 'nelder',
            'lbfgsb': 'lbfgsb'
        }
        actual_method = method_map.get(method, 'leastsq')

        # Create minimizer with complex-step Jacobian
        minimizer = Minimizer(
            DjordjevicSarkarModel.residual,
            params,
            fcn_args=(self.fit_data.f_ghz, self.fit_data.complex_eps),
            Dfun=self.jacobian_wrapper,
            scale_covar=True
        )

        # Run optimization
        result = minimizer.minimize(method=actual_method)

        # Extract fitted parameters
        fitted = FitParameters(
            eps_inf=float(result.params['eps_inf'].value),
            delta_eps=float(result.params['delta_eps'].value),
            omega1=float(result.params['omega1'].value),
            omega2=float(result.params['omega2'].value)
        )

        return fitted, result