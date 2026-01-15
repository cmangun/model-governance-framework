"""
Model Cards for Healthcare ML

FDA-ready model documentation following:
- FDA guidance on AI/ML-based SaMD
- Model Cards for Model Reporting (Mitchell et al.)
- NIST AI Risk Management Framework

Provides:
- Standardized model documentation
- Performance metrics by subgroup
- Intended use and limitations
- Ethical considerations
"""

from .generator import ModelCard, ModelCardGenerator, IntendedUse, PerformanceMetrics

__all__ = ["ModelCard", "ModelCardGenerator", "IntendedUse", "PerformanceMetrics"]
