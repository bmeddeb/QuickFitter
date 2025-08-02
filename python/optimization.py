"""
optimization.py - Optimization routines for dielectric fitting
"""
import numpy as np
from typing import Tuple, Any, Dict, Union
from lmfit import Parameters, Minimizer

from data_structures import FitData, FitParameters
from models import DjordjevicSarkarModel, HybridDebyeLorentzModel, ModelRegistry


class Optimizer:
    """Handles the optimization process for multiple models"""

    def __init__(self, fit_data: FitData, model_key: str = 'djordjevic_sarkar', model_config: Dict[str, Any] = None):
        self.fit_data = fit_data
        self.model_key = model_key
        self.model_config = model_config or {}
        self.model_class = ModelRegistry.get_model_class(model_key)

    def create_parameters_djordjevic_sarkar(self, initial: FitParameters) -> Parameters:
        """Create lmfit Parameters object with bounds for Djordjevic-Sarkar model"""
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

    def create_parameters_hybrid_debye_lorentz(self, initial_dict: Dict[str, float]) -> Parameters:
        """Create lmfit Parameters object with bounds for Hybrid Debye-Lorentz model"""
        params = Parameters()
        dk_data = np.real(self.fit_data.complex_eps)
        
        # Infer number of terms from the parameters
        n_terms = sum(1 for key in initial_dict if key.startswith('delta_eps_'))
        
        # eps_inf parameter
        params.add('eps_inf', value=initial_dict['eps_inf'], min=1.0, max=dk_data.max())
        
        # For each Debye-Lorentz term
        for i in range(n_terms):
            # Delta epsilon - dielectric strength
            params.add(f'delta_eps_{i}', value=initial_dict[f'delta_eps_{i}'], min=0.001, max=10)
            
            # Characteristic frequency - allow wide range
            params.add(f'f_k_{i}', value=initial_dict[f'f_k_{i}'], min=0.001, max=1000)
            
            # Conductivity term - allow small positive values
            params.add(f'sigma_k_{i}', value=initial_dict[f'sigma_k_{i}'], min=0.0001, max=1.0)
        
        return params

    def create_parameters(self, initial: Union[FitParameters, Dict[str, float]]) -> Parameters:
        """Create lmfit Parameters object with bounds based on model type"""
        if self.model_key == 'djordjevic_sarkar':
            return self.create_parameters_djordjevic_sarkar(initial)
        elif self.model_key == 'hybrid_debye_lorentz':
            return self.create_parameters_hybrid_debye_lorentz(initial)
        else:
            raise ValueError(f"Unsupported model: {self.model_key}")

    def jacobian_wrapper(self, params: Parameters, f_ghz: np.ndarray, complex_eps: np.ndarray) -> np.ndarray:
        """Wrapper to use complex-step Jacobian with lmfit"""
        return self.model_class.complex_step_jacobian(params.valuesdict(), f_ghz)

    def optimize(self, method: str = 'leastsq') -> Tuple[Union[FitParameters, Dict[str, float]], Any]:
        """Run optimization and return fitted parameters"""
        # Get initial guess based on model
        if self.model_key == 'djordjevic_sarkar':
            initial = self.model_class.estimate_initial_parameters(self.fit_data)
        elif self.model_key == 'hybrid_debye_lorentz':
            n_terms = self.model_config.get('n_terms', 2)
            initial = self.model_class.estimate_initial_parameters(self.fit_data, N=n_terms)
        else:
            raise ValueError(f"Unsupported model: {self.model_key}")

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
            self.model_class.residual,
            params,
            fcn_args=(self.fit_data.f_ghz, self.fit_data.complex_eps),
            Dfun=self.jacobian_wrapper,
            scale_covar=True
        )

        # Run optimization
        result = minimizer.minimize(method=actual_method)

        # Extract fitted parameters based on model type
        if self.model_key == 'djordjevic_sarkar':
            fitted = FitParameters(
                eps_inf=float(result.params['eps_inf'].value),
                delta_eps=float(result.params['delta_eps'].value),
                omega1=float(result.params['omega1'].value),
                omega2=float(result.params['omega2'].value)
            )
        elif self.model_key == 'hybrid_debye_lorentz':
            # Return as dictionary for hybrid model
            fitted = {}
            for param_name, param_obj in result.params.items():
                fitted[param_name] = float(param_obj.value)
        else:
            raise ValueError(f"Unsupported model: {self.model_key}")

        return fitted, result