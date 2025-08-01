"""
data_structures.py - Data classes and structures for dielectric fitting
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class FitData:
    """Container for all fitting data"""
    f_ghz: np.ndarray
    complex_eps: np.ndarray
    measured_dk: List[float] = field(default_factory=list)
    measured_df: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived properties"""
        if len(self.measured_dk) == 0:
            self.measured_dk = np.real(self.complex_eps).tolist()
        if len(self.measured_df) == 0:
            self.measured_df = (-np.imag(self.complex_eps) / np.real(self.complex_eps)).tolist()


@dataclass
class FitParameters:
    """Djordjevic-Sarkar model parameters"""
    eps_inf: float
    delta_eps: float
    omega1: float
    omega2: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'eps_inf': self.eps_inf,
            'delta_eps': self.delta_eps,
            'omega1': self.omega1,
            'omega2': self.omega2
        }

    @property
    def f1_ghz(self) -> float:
        return self.omega1 / (2 * np.pi * 1e9)

    @property
    def f2_ghz(self) -> float:
        return self.omega2 / (2 * np.pi * 1e9)


@dataclass
class MetricResult:
    """Stores the evaluation result for a single metric"""
    name: str
    score: float
    category: str
    value: float
    suggestion: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportEvaluation:
    """Stores the full evaluation for a report"""
    metrics: Dict[str, MetricResult]
    overall: MetricResult
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Return the evaluation as a markdown-formatted table"""
        headers = ["Metric", "Score", "Category", "Value", "Suggestion"]
        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join(["---"]*len(headers)) + " |"]
        for m in self.metrics.values():
            suggestion = m.suggestion or ""
            lines.append(f"| {m.name} | {m.score} | {m.category} | {m.value:.4g} | {suggestion} |")
        lines.append(
            f"| **Overall** | {self.overall.score} | {self.overall.category} | {self.overall.value} |  |"
        )
        if self.warnings:
            lines.append("\n**Warnings:**")
            for warning in self.warnings:
                lines.append(f"- {warning}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary format"""
        return {
            'overall_score': self.overall.score,
            'overall_category': self.overall.category,
            'metrics': {
                k: {
                    'score': v.score,
                    'category': v.category,
                    'value': v.value,
                    'suggestion': v.suggestion,
                    'details': v.details
                }
                for k, v in self.metrics.items()
            },
            'suggestions': self.suggestions,
            'warnings': self.warnings,
            'markdown_table': self.to_markdown()
        }