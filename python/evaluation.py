"""
evaluation.py - Fit quality evaluation
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional

from data_structures import FitData, FitParameters, MetricResult, ReportEvaluation
from models import DjordjevicSarkarModel


class FitEvaluator:
    """Evaluates the quality of a fit report based on configurable metrics"""

    DEFAULT_THRESHOLDS: Dict[str, Tuple[float, float, float, str]] = {
        "reduced_chi_square": (1e-4, 1e-3, 1e-2, "Revisit noise model or add a relaxation term."),
        "rms_real": (0.005, 0.010, 0.020, "Loosen bounds or refine initial ε_inf/Δε."),
        "rms_imag": (0.001, 0.002, 0.005, "Add extra pole or use a hybrid model for loss plateau."),
        "mean_residual": (1e-4, 5e-4, 1e-3, "Check for systematic bias at low frequencies."),
    }

    DEFAULT_WEIGHTS: Dict[str, int] = {
        'dpv': 1,
        'reduced_chi_square': 3,
        'rms_real': 2,
        'rms_imag': 2,
        'mean_residual': 1,
        'correlation': 2
    }

    def __init__(self, thresholds=None, weights=None):
        self.thresholds = thresholds or FitEvaluator.DEFAULT_THRESHOLDS
        self.weights = weights or FitEvaluator.DEFAULT_WEIGHTS

    def evaluate_from_fit_result(self, fit_data: FitData, params: FitParameters, result: Any) -> ReportEvaluation:
        """Evaluate fit quality from optimization result"""
        # Calculate residuals
        model_epsilon = DjordjevicSarkarModel.calculate(params.to_dict(), fit_data.f_ghz)
        real_residuals = np.real(model_epsilon) - np.real(fit_data.complex_eps)
        imag_residuals = np.imag(model_epsilon) - np.imag(fit_data.complex_eps)

        # Build report data
        n_dat = len(real_residuals) + len(imag_residuals)
        n_var = 4
        chi_sqr = np.sum(real_residuals**2) + np.sum(imag_residuals**2)
        red_chi_sqr = chi_sqr / (n_dat - n_var) if (n_dat - n_var) > 0 else 0

        # Extract correlations from fit result
        correlations = {}
        if hasattr(result, 'var_names') and hasattr(result, 'covar') and result.covar is not None:
            for i, name1 in enumerate(result.var_names):
                for j, name2 in enumerate(result.var_names):
                    if j > i:
                        corr_val = result.covar[i, j] / np.sqrt(result.covar[i, i] * result.covar[j, j])
                        if abs(corr_val) > 0.1:
                            correlations[(name1, name2)] = corr_val

        report_data = {
            'n_data_points': n_dat,
            'n_variables': n_var,
            'reduced_chi_square': red_chi_sqr,
            'rms_real': float(np.sqrt(np.mean(real_residuals**2))),
            'rms_imag': float(np.sqrt(np.mean(imag_residuals**2))),
            'mean_real': float(np.mean(real_residuals)),
            'mean_imag': float(np.mean(imag_residuals)),
            'correlations': correlations
        }

        return self.evaluate(report_data)

    def evaluate(self, report: Dict[str, Any]) -> ReportEvaluation:
        """Validates and evaluates a fit report dictionary"""
        required = {
            'n_data_points', 'n_variables', 'reduced_chi_square',
            'rms_real', 'rms_imag', 'mean_real', 'mean_imag', 'correlations'
        }
        missing = required - report.keys()
        if missing:
            raise KeyError(f"Missing required report fields: {missing}")

        self._validate_report(report)
        warnings = self._validate_physical_bounds(report)

        metrics: Dict[str, MetricResult] = {}
        suggestions: List[str] = []

        mean_res = max(abs(report['mean_real']), abs(report['mean_imag']))
        report_vals = dict(report)
        report_vals['mean_residual'] = mean_res

        # Data-points-per-variable
        dpv = report['n_data_points'] / report['n_variables']
        dpv_score, dpv_cat = self._evaluate_inverse_scalar(dpv, (10, 20, 30))
        dpv_sug = "Collect more data or simplify model." if dpv_cat == "Poor" else None
        metrics['dpv'] = MetricResult('dpv', dpv_score, dpv_cat, dpv, dpv_sug)
        if dpv_sug:
            suggestions.append(dpv_sug)

        # Scalar metrics
        for name, (g, e, gd, sug) in self.thresholds.items():
            val = report_vals.get(name, 0)
            sc, cat = self._evaluate_scalar(val, (g, e, gd))
            sugg = sug if cat == "Poor" else None
            metrics[name] = MetricResult(name, sc, cat, val, sugg)
            if sugg:
                suggestions.append(sugg)

        # Correlation
        corrs = report['correlations']
        if not corrs:
            corr_score, corr_cat, corr_sug = 100, "Low", None
            worst_pair, worst_val = None, 0.0
        else:
            worst_pair, worst_val = max(corrs.items(), key=lambda kv: abs(kv[1]))
            corr_score, corr_cat, corr_sug = self._evaluate_correlation(worst_val)

        corr_result = MetricResult(
            'correlation', corr_score, corr_cat,
            worst_val, corr_sug, details={'pair': worst_pair}
        )
        metrics['correlation'] = corr_result
        if corr_sug:
            suggestions.append(corr_sug)

        # Overall score
        total_weight = sum(self.weights.values())
        overall_score = sum(
            metrics[k].score * self.weights.get(k, 1)
            for k in metrics.keys()
            if k in self.weights
        ) / total_weight

        if overall_score >= 90:
            overall_cat = "Godly"
        elif overall_score >= 75:
            overall_cat = "Excellent"
        elif overall_score >= 60:
            overall_cat = "Good"
        else:
            overall_cat = "Poor"

        overall = MetricResult("overall", round(overall_score, 1), overall_cat, round(overall_score, 1))
        return ReportEvaluation(metrics=metrics, overall=overall, suggestions=suggestions, warnings=warnings)

    def _validate_report(self, report: Dict[str, Any]) -> None:
        """Validate report data types and values"""
        try:
            n_data = int(report['n_data_points'])
            n_vars = int(report['n_variables'])
            if n_vars <= 0:
                raise ValueError("n_variables must be positive")
            if n_data <= n_vars:
                raise ValueError("n_data_points must exceed n_variables")
            numeric_fields = ['reduced_chi_square', 'rms_real', 'rms_imag', 'mean_real', 'mean_imag']
            for field in numeric_fields:
                val = report.get(field, 0)
                if not isinstance(val, (int, float)) or math.isnan(val):
                    raise ValueError(f"{field} must be a valid number")
        except (TypeError, ValueError, KeyError) as e:
            raise ValueError(f"Invalid report data: {e}")

    def _validate_physical_bounds(self, report: Dict[str, Any]) -> List[str]:
        """Check if values are physically reasonable"""
        warnings = []
        if report.get('rms_real', 0) > 1.0:
            warnings.append("RMS real part unusually high - check data quality")
        if report.get('rms_imag', 0) > 0.1:
            warnings.append("RMS imaginary part very high - check loss data quality")
        rcs = report.get('reduced_chi_square', 0)
        if rcs < 1e-6 and rcs > 0:
            warnings.append("Reduced chi-square suspiciously low - possible overfitting")
        elif rcs > 1.0:
            warnings.append("Reduced chi-square > 1 - model may be inadequate")
        if report.get('rms_real', 0) < 0 or report.get('rms_imag', 0) < 0:
            warnings.append("Negative RMS values detected - calculation error")
        return warnings

    def _evaluate_scalar(self, value: float, thresholds: Tuple[float, float, float]) -> Tuple[int, str]:
        """Evaluates metrics where smaller values are better"""
        godly_max, excellent_max, good_max = thresholds
        if value < godly_max:
            return 100, "Godly"
        elif value < excellent_max:
            return 80, "Excellent"
        elif value < good_max:
            return 60, "Good"
        return 30, "Poor"

    def _evaluate_inverse_scalar(self, value: float, thresholds: Tuple[float, float, float]) -> Tuple[int, str]:
        """Evaluates metrics where larger values are better"""
        poor_min, good_min, excellent_min = thresholds
        if value >= excellent_min:
            return 100, "Godly"
        elif value >= good_min:
            return 80, "Excellent"
        elif value >= poor_min:
            return 60, "Good"
        return 30, "Poor"

    def _evaluate_correlation(self, corr_value: float) -> Tuple[int, str, Optional[str]]:
        """Evaluates correlation, returning score, category, and suggestion"""
        if not isinstance(corr_value, (int, float)) or math.isnan(corr_value):
            return 30, "Invalid", "Check correlation calculation"
        abs_corr = abs(corr_value)
        if abs_corr < 0.50:
            return 100, "Low", None
        elif abs_corr < 0.80:
            return 80, "Moderate", None
        elif abs_corr < 0.95:
            return 60, "High", "Inspect uncertainties; consider fixing one parameter."
        return 30, "Very High", "Fix or remove one highly correlated parameter."