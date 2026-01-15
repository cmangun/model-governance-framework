"""
Model Governance Framework

Production-grade AI governance for healthcare ML:
- Fairness and bias detection
- Model cards for FDA compliance
- Audit-ready documentation
"""
from src.fairness.evaluator import FairnessEvaluator
from src.fairness.bias_detector import BiasDetector, FairnessReport
from src.model_cards.generator import ModelCard, ModelCardGenerator

__version__ = "1.0.0"
__all__ = [
    "FairnessEvaluator",
    "BiasDetector", 
    "FairnessReport",
    "ModelCard",
    "ModelCardGenerator",
]
