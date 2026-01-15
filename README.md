# Model Governance Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive AI governance framework for regulated industries with bias detection, fairness metrics, and model documentation.

## Business Impact

- **FDA-ready** model documentation with automated model cards
- **Bias detection** across protected demographic attributes
- **Explainability** with SHAP and LIME integration
- **Zero compliance violations** across $51M+ portfolio

## Key Features

### Fairness and Bias Detection
- **Demographic Parity**: Equal positive rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Disparate Impact**: 4/5ths rule compliance
- **Intersectional Analysis**: Multi-attribute fairness

### Explainability
- SHAP value computation
- LIME explanations
- Feature importance analysis
- Decision boundary visualization

### Model Documentation
- Automated model cards (Google format)
- Intended use documentation
- Limitation and risk disclosure
- Healthcare-specific compliance notes

## Quick Start

```python
from src.fairness.bias_detector import BiasDetector, ProtectedAttribute

detector = BiasDetector()

report = detector.analyze(
    model_id="diabetes-classifier",
    model_version="1.0.0",
    predictions=[...],
    actuals=[...],
    protected_attributes={
        ProtectedAttribute.GENDER: ["male", "female", ...],
        ProtectedAttribute.AGE_GROUP: ["18-40", "41-65", "65+", ...],
    },
)

print(f"Overall Status: {report.overall_status}")
for violation in report.violations:
    print(f"  - {violation.metric}: {violation.description}")
```

## Project Structure

```
model-governance-framework/
├── src/
│   ├── fairness/
│   │   └── bias_detector.py     # Fairness metrics
│   ├── explainability/
│   │   └── explainer.py         # SHAP/LIME integration
│   ├── cards/
│   │   └── model_card.py        # Documentation generation
│   └── documentation/
│       └── report_generator.py  # Compliance reports
├── tests/
└── templates/
```

## Author

**Christopher Mangun** - [github.com/cmangun](https://github.com/cmangun)
