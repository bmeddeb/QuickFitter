"""
plotting.py - Plotting functionality for dielectric fitting
"""
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Any, Optional, Dict
from datetime import datetime

from data_structures import FitData, FitParameters, ReportEvaluation
from models import DjordjevicSarkarModel


class Plotter:
    """Handles all plotting functionality"""

    @staticmethod
    def create_plotly_plot(fit_data: FitData, params: FitParameters) -> str:
        """Create interactive Plotly plot"""
        # Calculate fitted model
        fitted_epsilon = DjordjevicSarkarModel.calculate(params.to_dict(), fit_data.f_ghz)

        # Create subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Real Permittivity', 'Loss Tangent'))

        # Real part
        fig.add_trace(
            go.Scatter(
                x=fit_data.f_ghz,
                y=np.real(fit_data.complex_eps),
                mode='markers',
                name='Measured Dk',
                marker=dict(color='black')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=fit_data.f_ghz,
                y=np.real(fitted_epsilon),
                mode='lines',
                name='Fitted Dk',
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Loss tangent
        measured_df = -np.imag(fit_data.complex_eps) / np.real(fit_data.complex_eps)
        fitted_df = -np.imag(fitted_epsilon) / np.real(fitted_epsilon)

        fig.add_trace(
            go.Scatter(
                x=fit_data.f_ghz,
                y=measured_df,
                mode='markers',
                name='Measured Df',
                marker=dict(color='black'),
                showlegend=False
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=fit_data.f_ghz,
                y=fitted_df,
                mode='lines',
                name='Fitted Df',
                line=dict(color='red'),
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Frequency (GHz)", row=1, col=1)
        fig.update_yaxes(title_text="Dielectric Constant (Dk)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (GHz)", row=2, col=1)
        fig.update_yaxes(title_text="Dissipation Factor (Df)", row=2, col=1)

        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )

        return pio.to_json(fig)

    @staticmethod
    def create_matplotlib_plot(fit_data: FitData, params: FitParameters) -> bytes:
        """Create downloadable matplotlib plot"""
        # Calculate fitted model
        fitted_epsilon = DjordjevicSarkarModel.calculate(params.to_dict(), fit_data.f_ghz)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Real part
        ax1.plot(fit_data.f_ghz, np.real(fit_data.complex_eps), 'ko', label='Measured')
        ax1.plot(fit_data.f_ghz, np.real(fitted_epsilon), 'r-', lw=2, label='Fitted')
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Dielectric Constant (Dk)')
        ax1.set_title('Real Permittivity')
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # Loss tangent
        measured_df = -np.imag(fit_data.complex_eps) / np.real(fit_data.complex_eps)
        fitted_df = -np.imag(fitted_epsilon) / np.real(fitted_epsilon)

        ax2.plot(fit_data.f_ghz, measured_df, 'ko', label='Measured')
        ax2.plot(fit_data.f_ghz, fitted_df, 'r-', lw=2, label='Fitted')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Dissipation Factor (Df)')
        ax2.set_title('Loss Tangent')
        ax2.legend()
        ax2.grid(True, alpha=0.5)

        plt.tight_layout()

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


class ReportGenerator:
    """Generates formatted reports"""

    @staticmethod
    def generate_text_report(fit_data: FitData, params: FitParameters, result: Any, evaluation: Optional[ReportEvaluation] = None) -> str:
        """Generate a text report"""
        # Calculate statistics
        model_epsilon = DjordjevicSarkarModel.calculate(params.to_dict(), fit_data.f_ghz)
        real_residuals = np.real(model_epsilon) - np.real(fit_data.complex_eps)
        imag_residuals = np.imag(model_epsilon) - np.imag(fit_data.complex_eps)

        n_dat = len(real_residuals) + len(imag_residuals)
        n_var = 4
        chi_sqr = np.sum(real_residuals**2) + np.sum(imag_residuals**2)
        red_chi_sqr = chi_sqr / (n_dat - n_var) if (n_dat - n_var) > 0 else 0

        # Extract correlations
        correlations_str = ""
        if hasattr(result, 'var_names') and hasattr(result, 'covar') and result.covar is not None:
            correlations_str = "\nCorrelations (from optimization):\n"
            for i, name1 in enumerate(result.var_names):
                for j, name2 in enumerate(result.var_names):
                    if j > i and result.covar is not None:
                        corr_val = result.covar[i, j] / np.sqrt(result.covar[i, i] * result.covar[j, j])
                        if abs(corr_val) > 0.1:
                            correlations_str += f"    ({name1}, {name2}) = {corr_val:+.4f}\n"

        report = f"""Djordjevic-Sarkar Fit Report
{'=' * 50}
Date: {datetime.now()}

Model and Parameters
--------------------------------------------------
ε'(ω) = ε_inf + (Δε / (2 * ln(ω₂/ω₁))) * ln((ω₂² + ω²) / (ω₁² + ω²))
ε''(ω) = -(Δε / ln(ω₂/ω₁)) * (atan(ω/ω₁) - atan(ω/ω₂))

Fitted parameters:
    eps_inf   = {params.eps_inf:.4f}
    delta_eps = {params.delta_eps:.4f}
    omega1    = {params.omega1:.4e} rad/s (f1 = {params.f1_ghz:.3f} GHz)
    omega2    = {params.omega2:.4e} rad/s (f2 = {params.f2_ghz:.3f} GHz)
    
    Frequency ratio: f2/f1 = {params.f2_ghz/params.f1_ghz:.1f}

Fit Statistics
    # data points = {n_dat}
    # variables   = {n_var}
    chi-square    = {chi_sqr:.4f}
    reduced chi-square = {red_chi_sqr:.4f}

Residual Analysis
    Real Part (Dk):
        Mean: {np.mean(real_residuals):.4f}
        Std Dev: {np.std(real_residuals):.4f}
        RMS: {np.sqrt(np.mean(real_residuals**2)):.4f}
    Imaginary Part (Loss Factor):
        Mean: {np.mean(imag_residuals):.4f}
        Std Dev: {np.std(imag_residuals):.4f}
        RMS: {np.sqrt(np.mean(imag_residuals**2)):.4f}

{correlations_str}"""

        if evaluation:
            report += f"\n\nFit Quality Evaluation\n{'-' * 50}\n"
            report += evaluation.to_markdown()

        return report

    @staticmethod
    def generate_json_report(fit_data: FitData, params: FitParameters, result: Any, evaluation: Optional[ReportEvaluation] = None) -> Dict[str, Any]:
        """Generate a JSON report"""
        # Calculate statistics
        model_epsilon = DjordjevicSarkarModel.calculate(params.to_dict(), fit_data.f_ghz)
        real_residuals = np.real(model_epsilon) - np.real(fit_data.complex_eps)
        imag_residuals = np.imag(model_epsilon) - np.imag(fit_data.complex_eps)

        n_dat = len(real_residuals) + len(imag_residuals)
        n_var = 4
        chi_sqr = np.sum(real_residuals**2) + np.sum(imag_residuals**2)
        red_chi_sqr = chi_sqr / (n_dat - n_var) if (n_dat - n_var) > 0 else 0

        # Extract correlation matrix
        corr_matrix = None
        if hasattr(result, 'var_names') and hasattr(result, 'covar') and result.covar is not None:
            n_params = len(result.var_names)
            corr_matrix = np.zeros((n_params, n_params))
            for i in range(n_params):
                for j in range(n_params):
                    if result.covar[i, i] > 0 and result.covar[j, j] > 0:
                        corr_matrix[i, j] = result.covar[i, j] / np.sqrt(result.covar[i, i] * result.covar[j, j])

        json_report = {
            "model": "Djordjevic-Sarkar",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "eps_inf": params.eps_inf,
                "delta_eps": params.delta_eps,
                "omega1": params.omega1,
                "omega2": params.omega2,
                "f1_ghz": params.f1_ghz,
                "f2_ghz": params.f2_ghz
            },
            "fit_statistics": {
                "n_data_points": n_dat,
                "n_variables": n_var,
                "chi_square": chi_sqr,
                "reduced_chi_square": red_chi_sqr
            },
            "residual_analysis": {
                "real_part": {
                    "mean": float(np.mean(real_residuals)),
                    "std_dev": float(np.std(real_residuals)),
                    "rms": float(np.sqrt(np.mean(real_residuals**2)))
                },
                "imaginary_part": {
                    "mean": float(np.mean(imag_residuals)),
                    "std_dev": float(np.std(imag_residuals)),
                    "rms": float(np.sqrt(np.mean(imag_residuals**2)))
                }
            },
            "correlation_matrix": corr_matrix.tolist() if corr_matrix is not None else None
        }

        if evaluation:
            json_report['evaluation'] = evaluation.to_dict()

        return json_report