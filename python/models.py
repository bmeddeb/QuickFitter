"""
models.py - Djordjevic-Sarkar dielectric model implementation
"""
import numpy as np
import pandas as pd
import io
from typing import Dict, Any
from lmfit import Parameters

# Import from local modules (will be available after loading)
from data_structures import FitData, FitParameters

# Try to import js for console logging in Pyodide
try:
    import js
    HAS_JS = True
except ImportError:
    HAS_JS = False


class DataLoader:
    """Handles data loading and parsing"""

    @staticmethod
    def load_from_csv(csv_content: str) -> FitData:
        """Load data from CSV content"""
        try:
            data = pd.read_csv(io.StringIO(csv_content)).dropna()
            f_ghz = data.iloc[:, 0].values
            dk = data.iloc[:, 1].values
            df = data.iloc[:, 2].values
            complex_epsilon = dk - 1j * (dk * df)

            return FitData(
                f_ghz=f_ghz,
                complex_eps=complex_epsilon,
                measured_dk=dk.tolist(),
                measured_df=df.tolist()
            )
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")


class DjordjevicSarkarModel:
    """Djordjevic-Sarkar dielectric model implementation"""

    @staticmethod
    def calculate(params: Dict[str, Any], f_ghz: np.ndarray) -> np.ndarray:
        """Calculate complex permittivity from model parameters"""
        omega = 2 * np.pi * f_ghz * 1e9

        # Extract real part of parameters (in case of complex-step)
        eps_inf = np.real(params['eps_inf'])
        delta_eps = np.real(params['delta_eps'])
        omega1 = np.real(params['omega1'])
        omega2 = np.real(params['omega2'])

        # Check if any parameter has imaginary part (complex-step)
        if any(np.iscomplex(params[k]) for k in params):
            # We're in complex-step mode, use complex arithmetic
            eps_inf = params['eps_inf']
            delta_eps = params['delta_eps']
            omega1 = params['omega1']
            omega2 = params['omega2']

        with np.errstate(divide='ignore', invalid='ignore'):
            if np.real(omega2) <= 0 or np.real(omega1) <= 0 or np.real(omega2) <= np.real(omega1):
                return np.full_like(f_ghz, np.nan, dtype=complex)

            log_term = np.log((omega2**2 + omega**2) / (omega1**2 + omega**2))
            eps_prime = eps_inf + (delta_eps / (2 * np.log(omega2 / omega1))) * log_term

            atan_term = np.arctan(omega / omega1) - np.arctan(omega / omega2)
            eps_double_prime = -(delta_eps / np.log(omega2 / omega1)) * atan_term

            return eps_prime + 1j * eps_double_prime

    @staticmethod
    def residual(params: Parameters, f_ghz: np.ndarray, complex_epsilon_data: np.ndarray) -> np.ndarray:
        """Calculate residuals for optimization"""
        p_dict = params.valuesdict()
        model_epsilon = DjordjevicSarkarModel.calculate(p_dict, f_ghz)

        real_residual = (np.real(model_epsilon) - np.real(complex_epsilon_data))
        imag_residual = (np.imag(model_epsilon) - np.imag(complex_epsilon_data))
        return np.concatenate([real_residual, imag_residual])

    @staticmethod
    def complex_step_jacobian(params_dict: Dict[str, float], f_ghz: np.ndarray, h: float = 1e-20) -> np.ndarray:
        """
        Calculate Jacobian using complex-step differentiation.
        This is numerically exact to machine precision for small h.
        """
        names = list(params_dict.keys())
        n = len(names)

        # Calculate base model
        base = DjordjevicSarkarModel.calculate(params_dict, f_ghz)
        base_vec = np.concatenate([base.real, base.imag])

        # Initialize Jacobian
        J = np.zeros((base_vec.size, n))

        # Calculate derivatives using complex step
        for i, name in enumerate(names):
            p2 = params_dict.copy()
            p2[name] += h * 1j  # Add imaginary perturbation

            # Calculate perturbed model
            y2 = DjordjevicSarkarModel.calculate(p2, f_ghz)
            vec2 = np.concatenate([y2.real, y2.imag])

            # Extract derivative from imaginary part
            J[:, i] = np.imag(vec2) / h

        return J

    @staticmethod
    def estimate_initial_parameters(fit_data: FitData, pct: float = 0.05) -> FitParameters:
        """
        Robust initial guess for model parameters.

        Args:
            fit_data: FitData object containing frequency and permittivity
            pct: fraction of points for averaging at ends
        """
        # Sort data by frequency
        idx = np.argsort(fit_data.f_ghz)
        f = fit_data.f_ghz[idx]
        eps_real = np.real(fit_data.complex_eps)[idx]
        eps_imag = -np.imag(fit_data.complex_eps)[idx]  # Note: loss is negative imaginary part

        n = len(f)
        k = max(1, int(n * pct))

        # Estimate parameters
        eps_s = np.mean(eps_real[:k])        # static permittivity at low freq
        eps_inf = np.mean(eps_real[-k:])     # high-freq permittivity
        delta_eps = max(eps_s - eps_inf, 1e-6)

        # Find loss peak - looking at the actual loss (imaginary part)
        loss_tangent = eps_imag / eps_real  # This is Df
        peak_idx = np.argmax(loss_tangent)
        f_peak = f[peak_idx]

        # For Djordjevic-Sarkar model, omega1 and omega2 define the relaxation band
        # omega1 should be below the main relaxation, omega2 should be well above it

        # Start with frequency range of data
        f_min = f[0]
        f_max = f[-1]

        # Use JavaScript console for logging in Pyodide
        if HAS_JS:
            js.console.log(f"Data frequency range: {f_min:.3f} - {f_max:.3f} GHz")
            js.console.log(f"Loss peak at: {f_peak:.3f} GHz")

        # Set omega1 to be at or below the lowest measured frequency
        f_low = f_min * 0.5  # Half the minimum frequency

        # Set omega2 to be well above the highest measured frequency
        # The factor here is critical - it needs to be high enough
        f_high = f_max * 20  # 20x the maximum frequency

        # Ensure minimum values
        f_low = max(0.1, f_low)      # At least 0.1 GHz
        f_high = max(500, f_high)    # At least 500 GHz

        # If we have a clear loss peak, use it to refine estimates
        if f_peak > 0 and loss_tangent[peak_idx] > 0.001:  # Significant loss peak
            # omega1 should be well below the peak
            f_low = min(f_low, f_peak * 0.01)  # 1/100th of peak
            # omega2 should be well above the peak
            f_high = max(f_high, f_peak * 100)  # 100x peak

        # Convert to rad/s
        omega1 = 2 * np.pi * f_low * 1e9
        omega2 = 2 * np.pi * f_high * 1e9

        # Safety check
        if omega2 <= omega1:
            omega2 = omega1 * 1000

        # Use JavaScript console for logging in Pyodide
        if HAS_JS:
            js.console.log(f"Initial frequency estimates: f1={f_low:.3f} GHz, f2={f_high:.3f} GHz")
            js.console.log(f"Initial omega estimates: omega1={omega1:.3e} rad/s, omega2={omega2:.3e} rad/s")
            js.console.log(f"Initial parameter estimates: eps_inf={eps_inf:.4f}, delta_eps={delta_eps:.4f}")

        return FitParameters(
            eps_inf=float(eps_inf),
            delta_eps=float(delta_eps),
            omega1=float(omega1),
            omega2=float(omega2)
        )


class HybridDebyeLorentzModel:
    """
    Hybrid Debye-Lorentz dielectric model implementation.

    This class provides static methods to calculate the complex permittivity
    based on a hybrid model, estimate initial parameters, and compute residuals
    and Jacobians for optimization routines like lmfit.

    Formula Note:
    The model calculates the complex permittivity ε*(f) using the formula:
    ε*(f) = ε_inf + Σ [Δε_k / (1 + j(f/f_k)) + (j * f * σ_k) / (f² + f_k²)]
    where the summation is from k=1 to N.

    Initial Parameters Note:
    - ε_inf: Estimated as 80% of the minimum measured dielectric constant.
    - Δε_k: The total range of the dielectric constant is split evenly among the N terms.
    - f_k: Frequencies are logarithmically spaced across the measurement range.
    - σ_k: Initialized to a small constant value (0.01).
    """

    @staticmethod
    def calculate(params: Dict[str, Any], f_ghz: np.ndarray) -> np.ndarray:
        """
        Calculate complex permittivity from hybrid model parameters.
        The number of terms, N, is inferred from the 'params' dictionary.
        """
        # Infer the number of terms, N, from the parameters
        n_terms = sum(1 for key in params if key.startswith('delta_eps_'))

        # Extract parameters
        eps_inf = params['eps_inf']
        delta_eps = np.array([params[f'delta_eps_{i}'] for i in range(n_terms)])
        f_k = np.array([params[f'f_k_{i}'] for i in range(n_terms)])
        sigma_k = np.array([params[f'sigma_k_{i}'] for i in range(n_terms)])

        # Ensure we're using real parts unless it's a complex-step differentiation
        if not any(np.iscomplex(v) for v in params.values()):
            eps_inf = np.real(eps_inf)
            delta_eps = np.real(delta_eps)
            f_k = np.real(f_k)
            sigma_k = np.real(sigma_k)

        # Use frequency in GHz directly, as per the original model formulation
        F = f_ghz.reshape(-1, 1)  # shape (M, 1)
        FK = f_k.reshape(1, -1)    # shape (1, N)

        # --- Calculate model components ---
        # To avoid division by zero or other errors if parameters are invalid
        with np.errstate(divide='ignore', invalid='ignore'):
            # Debye relaxation terms
            debye_denominator = 1 + 1j * (F / FK)
            debye_terms = delta_eps / debye_denominator

            # Modified Lorentz conductivity terms
            lorentz_denominator = F**2 + FK**2
            # Handle potential zeros in the denominator
            lorentz_denominator[lorentz_denominator == 0] = np.inf
            lorentz_terms = (1j * F * sigma_k) / lorentz_denominator

            # Sum the components for the final complex permittivity
            # np.sum with axis=1 collapses the N terms for each frequency point
            complex_epsilon = eps_inf + np.sum(debye_terms, axis=1) + np.sum(lorentz_terms, axis=1)

        # Replace any NaNs that might have occurred with a large number
        # to guide the fitter away from invalid parameter regions.
        complex_epsilon[np.isnan(complex_epsilon)] = 1e9 + 1j*1e9
        return complex_epsilon

    @staticmethod
    def residual(params: Parameters, f_ghz: np.ndarray, complex_epsilon_data: np.ndarray) -> np.ndarray:
        """Calculate residuals for optimization."""
        p_dict = params.valuesdict()
        model_epsilon = HybridDebyeLorentzModel.calculate(p_dict, f_ghz)

        # Calculate residuals for real and imaginary parts separately
        real_residual = np.real(model_epsilon) - np.real(complex_epsilon_data)
        imag_residual = np.imag(model_epsilon) - np.imag(complex_epsilon_data)
        
        # Concatenate into a single residual vector for the optimizer
        return np.concatenate([real_residual, imag_residual])

    @staticmethod
    def complex_step_jacobian(params_dict: Dict[str, float], f_ghz: np.ndarray, h: float = 1e-20) -> np.ndarray:
        """
        Calculate Jacobian using complex-step differentiation.
        This is numerically exact to machine precision for a small step 'h'.
        """
        names = list(params_dict.keys())
        n_params = len(names)

        # Calculate base model output vector (concatenated real and imag parts)
        base_epsilon = HybridDebyeLorentzModel.calculate(params_dict, f_ghz)
        base_vec = np.concatenate([base_epsilon.real, base_epsilon.imag])

        # Initialize Jacobian matrix
        J = np.zeros((base_vec.size, n_params))

        # Calculate derivatives for each parameter using the complex step method
        for i, name in enumerate(names):
            p_perturbed = params_dict.copy()
            p_perturbed[name] += h * 1j  # Add a small imaginary perturbation

            # Calculate the model output with the perturbed parameter
            epsilon_perturbed = HybridDebyeLorentzModel.calculate(p_perturbed, f_ghz)
            vec_perturbed = np.concatenate([epsilon_perturbed.real, epsilon_perturbed.imag])

            # The derivative is the imaginary part of the result divided by the step size
            J[:, i] = np.imag(vec_perturbed) / h

        return J

    @staticmethod
    def estimate_initial_parameters(fit_data: FitData, N: int = 2) -> Dict[str, float]:
        """
        Generates robust initial guesses for the Hybrid Model parameters.

        Args:
            fit_data: A data object containing frequency and complex permittivity arrays.
            N: The number of Debye-Lorentz terms to use in the model.

        Returns:
            A dictionary of initial parameter names and their estimated values.
        """
        # Sort data by frequency to ensure correct range calculations
        idx = np.argsort(fit_data.f_ghz)
        freq = fit_data.f_ghz[idx]
        dk_exp = np.real(fit_data.complex_eps)[idx]

        # Calculate frequency and dielectric constant ranges from experimental data
        dk_range = np.max(dk_exp) - np.min(dk_exp)
        freq_min = np.min(freq)
        freq_max = np.max(freq)
        freq_mean = np.mean(freq)

        # --- Estimate initial values based on the model's logic ---

        # 1. High-frequency permittivity (eps_inf)
        eps_inf_initial = np.min(dk_exp) * 0.8

        # 2. Debye strengths (delta_eps_k) - split the dielectric range
        delta_eps_initial = max(1e-6, dk_range / N)

        # 3. Characteristic frequencies (f_k) - spread across measurement range
        if N == 1:
            f_k_initial = np.array([freq_mean])
        else:
            # Logarithmically space the frequencies across a slightly expanded range
            f_k_initial = np.logspace(
                np.log10(max(1e-3, freq_min / 2)), # Avoid log(0)
                np.log10(freq_max * 2),
                N
            )

        # 4. Conductivity terms (sigma_k) - start small
        sigma_k_initial = 0.01

        # --- Assemble the parameters into a dictionary ---
        initial_params = {'eps_inf': float(eps_inf_initial)}
        for i in range(N):
            initial_params[f'delta_eps_{i}'] = float(delta_eps_initial)
            initial_params[f'f_k_{i}'] = float(f_k_initial[i])
            initial_params[f'sigma_k_{i}'] = float(sigma_k_initial)

        return initial_params


class ModelRegistry:
    """Registry for available dielectric models"""
    
    _models = {
        'djordjevic_sarkar': {
            'name': 'Djordjevic-Sarkar',
            'description': 'Wideband model for frequency-dependent dielectric materials',
            'class': DjordjevicSarkarModel,
            'parameters': ['eps_inf', 'delta_eps', 'omega1', 'omega2'],
            'parameter_names': ['ε∞', 'Δε', 'ω₁ (rad/s)', 'ω₂ (rad/s)']
        },
        'hybrid_debye_lorentz': {
            'name': 'Hybrid Debye-Lorentz',
            'description': 'Combined Debye relaxation and Lorentz resonance model',
            'class': HybridDebyeLorentzModel,
            'parameters': 'variable',  # Depends on N terms
            'parameter_names': 'variable'
        }
    }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available models"""
        return cls._models.copy()
    
    @classmethod
    def get_model(cls, model_key: str) -> Dict[str, Any]:
        """Get specific model information"""
        if model_key not in cls._models:
            raise ValueError(f"Model '{model_key}' not found. Available: {list(cls._models.keys())}")
        return cls._models[model_key].copy()
    
    @classmethod
    def get_model_class(cls, model_key: str):
        """Get the model class for a specific model"""
        return cls.get_model(model_key)['class']