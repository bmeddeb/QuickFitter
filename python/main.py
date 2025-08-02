"""
main.py - Main API class and legacy function wrappers
"""
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

# Import from local modules
from data_structures import FitData, FitParameters
from models import DataLoader, DjordjevicSarkarModel, HybridDebyeLorentzModel, ModelRegistry
from optimization import Optimizer
from evaluation import FitEvaluator
from plotting import Plotter, ReportGenerator

# Try to import js for console logging
try:
    import js
    HAS_JS = True
except ImportError:
    HAS_JS = False


class DielectricFitter:
    """Main class that orchestrates the fitting process for multiple models"""

    def __init__(self, model_key: str = 'djordjevic_sarkar', model_config: Dict[str, Any] = None):
        self.fit_data: Optional[FitData] = None
        self.current_params: Optional[Union[FitParameters, Dict[str, float]]] = None
        self.last_result: Optional[Any] = None
        self.model_key = model_key
        self.model_config = model_config or {}
        self.model_class = ModelRegistry.get_model_class(model_key)

    def load_data(self, csv_content: str) -> Dict[str, Any]:
        """Load data from CSV"""
        try:
            self.fit_data = DataLoader.load_from_csv(csv_content)
            return {
                "success": True,
                "message": "Data loaded successfully",
                "n_points": len(self.fit_data.f_ghz)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_analysis(self, method: str = 'leastsq') -> Dict[str, Any]:
        """Run the full analysis pipeline"""
        if self.fit_data is None:
            return {"error": "No data loaded"}

        try:
            # Run optimization with model selection
            optimizer = Optimizer(self.fit_data, self.model_key, self.model_config)
            self.current_params, self.last_result = optimizer.optimize(method)

            # Evaluate fit
            evaluator = FitEvaluator()
            evaluation = evaluator.evaluate_from_fit_result(
                self.fit_data,
                self.current_params,
                self.last_result,
                self.model_key
            )

            # Generate reports
            text_report = ReportGenerator.generate_text_report(
                self.fit_data,
                self.current_params,
                self.last_result,
                evaluation,
                self.model_key
            )
            json_report = ReportGenerator.generate_json_report(
                self.fit_data,
                self.current_params,
                self.last_result,
                evaluation,
                self.model_key
            )

            # Create plot
            plot_json = Plotter.create_plotly_plot(self.fit_data, self.current_params, self.model_key)

            # Handle different parameter formats
            if isinstance(self.current_params, FitParameters):
                fitted_params_dict = self.current_params.to_dict()
            else:
                fitted_params_dict = self.current_params

            return {
                "report": text_report,
                "json_data": json_report,
                "plot_json": plot_json,
                "fitted_params": fitted_params_dict,
                "f_ghz": self.fit_data.f_ghz.tolist(),
                "measured_dk": self.fit_data.measured_dk,
                "measured_df": self.fit_data.measured_df,
                "model_key": self.model_key
            }

        except Exception as e:
            if HAS_JS:
                js.console.error(f"Analysis failed: {e}")
            return {"error": str(e)}

    def calculate_model_from_params(self, eps_inf: float, delta_eps: float,
                                    omega1: float, omega2: float) -> Dict[str, Any]:
        """Calculate model response for given parameters"""
        if self.fit_data is None:
            return {"error": "No data loaded"}

        params = {
            'eps_inf': eps_inf,
            'delta_eps': delta_eps,
            'omega1': omega1,
            'omega2': omega2
        }

        model_epsilon = DjordjevicSarkarModel.calculate(params, self.fit_data.f_ghz)

        return {
            "eps_prime": np.real(model_epsilon).tolist(),
            "eps_double_prime": np.imag(model_epsilon).tolist()
        }

    def create_updated_report(self, eps_inf: float, delta_eps: float,
                              omega1: float, omega2: float) -> Dict[str, Any]:
        """Create a report for manually adjusted parameters"""
        if self.fit_data is None:
            return {"error": "No data loaded"}

        # Create parameter object
        params = FitParameters(eps_inf, delta_eps, omega1, omega2)

        # Create a dummy result object for report generation
        class DummyResult:
            var_names = None
            covar = None

        dummy_result = DummyResult()

        # Generate reports
        text_report = ReportGenerator.generate_text_report(
            self.fit_data,
            params,
            dummy_result
        )
        json_report = ReportGenerator.generate_json_report(
            self.fit_data,
            params,
            dummy_result
        )

        return {
            "report": text_report,
            "json_data": json_report
        }

    def evaluate_current_fit(self, eps_inf: float, delta_eps: float,
                             omega1: float, omega2: float) -> Dict[str, Any]:
        """Evaluate the quality of current parameters"""
        if self.fit_data is None:
            return {"error": "No data loaded"}

        params = FitParameters(eps_inf, delta_eps, omega1, omega2)

        # Use last result if available, otherwise create dummy
        result = self.last_result if self.last_result is not None else type('obj', (object,), {'var_names': None, 'covar': None})

        # Evaluate
        evaluator = FitEvaluator()
        evaluation = evaluator.evaluate_from_fit_result(self.fit_data, params, result, self.model_key)

        # Generate JSON report
        json_report = ReportGenerator.generate_json_report(
            self.fit_data,
            params,
            result,
            evaluation,
            self.model_key
        )

        return json_report

    def create_downloadable_plot(self, eps_inf: float, delta_eps: float,
                                 omega1: float, omega2: float) -> Optional[bytes]:
        """Create a downloadable PNG plot"""
        if self.fit_data is None:
            return None

        params = FitParameters(eps_inf, delta_eps, omega1, omega2)
        return Plotter.create_matplotlib_plot(self.fit_data, params, self.model_key)

    def get_measured_data(self) -> Dict[str, Any]:
        """Return the stored measured data"""
        if self.fit_data is None:
            return {
                "measured_dk": [],
                "measured_df": []
            }

        return {
            "measured_dk": self.fit_data.measured_dk,
            "measured_df": self.fit_data.measured_df
        }


# ============================================================================
# Wrapper functions for backward compatibility
# ============================================================================

# Create a global fitter instance (backward compatibility with Djordjevic-Sarkar)
_fitter = DielectricFitter(model_key='djordjevic_sarkar')

# Create model-specific instances
_hybrid_fitter = DielectricFitter(model_key='hybrid_debye_lorentz', model_config={'n_terms': 2})

def load_data(csv_content: str) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper for data loading"""
    result = _fitter.load_data(csv_content)
    if result.get("success"):
        return _fitter.fit_data.f_ghz, _fitter.fit_data.complex_eps
    else:
        raise ValueError(result.get("error", "Unknown error"))

def run_analysis(csv_content: str, method: str = 'leastsq') -> str:
    """Legacy wrapper for running analysis"""
    try:
        load_result = _fitter.load_data(csv_content)
        if not load_result.get("success"):
            return json.dumps({"error": load_result.get("error", "Failed to load data")})
        result = _fitter.run_analysis(method)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def calculate_model_from_params(eps_inf: float, delta_eps: float,
                                omega1: float, omega2: float) -> str:
    """Legacy wrapper for model calculation"""
    result = _fitter.calculate_model_from_params(eps_inf, delta_eps, omega1, omega2)
    return json.dumps(result)

def create_updated_report(eps_inf: float, delta_eps: float,
                          omega1: float, omega2: float) -> str:
    """Legacy wrapper for report creation"""
    result = _fitter.create_updated_report(eps_inf, delta_eps, omega1, omega2)
    return json.dumps(result)

def evaluate_current_fit(eps_inf: float, delta_eps: float,
                         omega1: float, omega2: float) -> str:
    """Legacy wrapper for fit evaluation"""
    result = _fitter.evaluate_current_fit(eps_inf, delta_eps, omega1, omega2)
    return json.dumps(result)

def create_downloadable_plot(eps_inf: float, delta_eps: float,
                             omega1: float, omega2: float) -> Optional[bytes]:
    """Legacy wrapper for plot creation"""
    return _fitter.create_downloadable_plot(eps_inf, delta_eps, omega1, omega2)

def get_measured_data() -> str:
    """Legacy wrapper for getting measured data"""
    result = _fitter.get_measured_data()
    return json.dumps(result)

def estimate_initial_parameters(f_ghz: np.ndarray, complex_epsilon: np.ndarray, pct: float = 0.05) -> Dict[str, float]:
    """Legacy wrapper for initial parameter estimation"""
    fit_data = FitData(f_ghz, complex_epsilon)
    params = DjordjevicSarkarModel.estimate_initial_parameters(fit_data, pct)
    return params.to_dict()


# ============================================================================
# New API functions for model selection
# ============================================================================

def get_available_models() -> str:
    """Get available dielectric models"""
    models = ModelRegistry.get_available_models()
    return json.dumps(models)

def run_analysis_with_model(csv_content: str, model_key: str = 'djordjevic_sarkar', 
                           model_config: Dict[str, Any] = None, method: str = 'leastsq') -> str:
    """Run analysis with specified model"""
    try:
        if HAS_JS:
            js.console.log(f"run_analysis_with_model called with model_key: {model_key}")
            js.console.log(f"model_config: {model_config}")
        
        fitter = DielectricFitter(model_key, model_config or {})
        load_result = fitter.load_data(csv_content)
        if not load_result.get("success"):
            return json.dumps({"error": load_result.get("error", "Failed to load data")})
        result = fitter.run_analysis(method)
        return json.dumps(result)
    except Exception as e:
        if HAS_JS:
            js.console.error(f"Error in run_analysis_with_model: {e}")
        import traceback
        return json.dumps({"error": f"{str(e)}\n{traceback.format_exc()}"})

def calculate_model_from_params_generic(model_key: str, params_dict: Dict[str, float], 
                                       f_ghz_list: list) -> str:
    """Calculate model response for generic model with arbitrary parameters"""
    try:
        model_class = ModelRegistry.get_model_class(model_key)
        f_ghz = np.array(f_ghz_list)
        model_epsilon = model_class.calculate(params_dict, f_ghz)
        
        return json.dumps({
            "eps_prime": np.real(model_epsilon).tolist(),
            "eps_double_prime": np.imag(model_epsilon).tolist()
        })
    except Exception as e:
        return json.dumps({"error": str(e)})