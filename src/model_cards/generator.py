"""
Model Card Generator for Healthcare ML

Generates standardized model documentation for:
- FDA 510(k)/De Novo submissions
- Audit and compliance requirements
- Clinical deployment documentation
- Transparency and accountability
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Classification of model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    NLP = "natural_language_processing"
    MULTIMODAL = "multimodal"
    GENERATIVE = "generative"


class RiskLevel(str, Enum):
    """FDA risk classification for SaMD."""
    CLASS_I = "class_i"
    CLASS_II = "class_ii"
    CLASS_III = "class_iii"
    NON_DEVICE = "non_device"


class ClinicalSetting(str, Enum):
    """Healthcare deployment settings."""
    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    EMERGENCY = "emergency"
    PRIMARY_CARE = "primary_care"
    SPECIALTY = "specialty"
    HOME = "home_health"
    LABORATORY = "laboratory"
    RADIOLOGY = "radiology"


@dataclass
class IntendedUse:
    """Model intended use specification."""
    primary_use: str
    target_population: str
    clinical_setting: list[ClinicalSetting]
    user_types: list[str]
    out_of_scope_uses: list[str]
    regulatory_status: RiskLevel
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_use": self.primary_use,
            "target_population": self.target_population,
            "clinical_setting": [s.value for s in self.clinical_setting],
            "user_types": self.user_types,
            "out_of_scope_uses": self.out_of_scope_uses,
            "regulatory_status": self.regulatory_status.value,
        }


@dataclass 
class TrainingData:
    """Training data documentation."""
    source_description: str
    collection_methodology: str
    preprocessing_steps: list[str]
    sample_size: int
    date_range: tuple[datetime, datetime]
    demographics: dict[str, dict[str, int]]
    exclusion_criteria: list[str]
    known_limitations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_description": self.source_description,
            "collection_methodology": self.collection_methodology,
            "preprocessing_steps": self.preprocessing_steps,
            "sample_size": self.sample_size,
            "date_range": {
                "start": self.date_range[0].isoformat(),
                "end": self.date_range[1].isoformat(),
            },
            "demographics": self.demographics,
            "exclusion_criteria": self.exclusion_criteria,
            "known_limitations": self.known_limitations,
        }


@dataclass
class PerformanceMetrics:
    """Model performance documentation."""
    
    # Overall metrics
    overall_accuracy: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    overall_auc_roc: float | None = None
    overall_auc_pr: float | None = None
    
    # Subgroup performance
    subgroup_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Clinical metrics
    sensitivity: float | None = None
    specificity: float | None = None
    ppv: float | None = None
    npv: float | None = None
    
    # Confidence intervals
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    
    # Evaluation dataset info
    eval_dataset_size: int = 0
    eval_date: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": {
                "accuracy": round(self.overall_accuracy, 4),
                "precision": round(self.overall_precision, 4),
                "recall": round(self.overall_recall, 4),
                "f1_score": round(self.overall_f1, 4),
                "auc_roc": round(self.overall_auc_roc, 4) if self.overall_auc_roc else None,
                "auc_pr": round(self.overall_auc_pr, 4) if self.overall_auc_pr else None,
            },
            "clinical": {
                "sensitivity": round(self.sensitivity, 4) if self.sensitivity else None,
                "specificity": round(self.specificity, 4) if self.specificity else None,
                "positive_predictive_value": round(self.ppv, 4) if self.ppv else None,
                "negative_predictive_value": round(self.npv, 4) if self.npv else None,
            },
            "subgroup_metrics": self.subgroup_metrics,
            "confidence_intervals": {
                k: {"lower": v[0], "upper": v[1]} 
                for k, v in self.confidence_intervals.items()
            },
            "evaluation_info": {
                "dataset_size": self.eval_dataset_size,
                "evaluation_date": self.eval_date.isoformat(),
            },
        }


@dataclass
class EthicalConsiderations:
    """Ethical and fairness documentation."""
    
    potential_biases: list[str]
    mitigation_strategies: list[str]
    fairness_evaluations_performed: list[str]
    demographic_groups_analyzed: list[str]
    disparate_impact_assessment: str
    human_oversight_requirements: list[str]
    explainability_approach: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "potential_biases": self.potential_biases,
            "mitigation_strategies": self.mitigation_strategies,
            "fairness_evaluations": self.fairness_evaluations_performed,
            "demographic_groups_analyzed": self.demographic_groups_analyzed,
            "disparate_impact_assessment": self.disparate_impact_assessment,
            "human_oversight_requirements": self.human_oversight_requirements,
            "explainability_approach": self.explainability_approach,
        }


@dataclass
class ModelCard:
    """
    Complete Model Card for Healthcare ML.
    
    Follows FDA guidance and Model Cards for Model Reporting standard.
    """
    
    # Model identity
    model_id: str
    model_name: str
    model_version: str
    model_type: ModelType
    card_id: str = field(default_factory=lambda: f"mc_{uuid.uuid4().hex[:12]}")
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Core sections
    description: str = ""
    intended_use: IntendedUse | None = None
    training_data: TrainingData | None = None
    performance: PerformanceMetrics | None = None
    ethical_considerations: EthicalConsiderations | None = None
    
    # Technical details
    architecture: str = ""
    framework: str = ""
    input_format: str = ""
    output_format: str = ""
    hardware_requirements: str = ""
    
    # Limitations
    limitations: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    
    # Ownership
    developers: list[str] = field(default_factory=list)
    organization: str = ""
    contact: str = ""
    license: str = ""
    
    # References
    citation: str = ""
    references: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "card_id": self.card_id,
            "model_identity": {
                "model_id": self.model_id,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_type": self.model_type.value,
            },
            "timestamps": {
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
            },
            "description": self.description,
            "intended_use": self.intended_use.to_dict() if self.intended_use else None,
            "training_data": self.training_data.to_dict() if self.training_data else None,
            "performance": self.performance.to_dict() if self.performance else None,
            "ethical_considerations": self.ethical_considerations.to_dict() if self.ethical_considerations else None,
            "technical_details": {
                "architecture": self.architecture,
                "framework": self.framework,
                "input_format": self.input_format,
                "output_format": self.output_format,
                "hardware_requirements": self.hardware_requirements,
            },
            "limitations": {
                "known_limitations": self.limitations,
                "failure_modes": self.failure_modes,
                "caveats": self.caveats,
            },
            "ownership": {
                "developers": self.developers,
                "organization": self.organization,
                "contact": self.contact,
                "license": self.license,
            },
            "references": {
                "citation": self.citation,
                "additional_references": self.references,
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = [
            f"# Model Card: {self.model_name}",
            "",
            f"**Model ID:** {self.model_id}  ",
            f"**Version:** {self.model_version}  ",
            f"**Type:** {self.model_type.value}  ",
            f"**Last Updated:** {self.last_updated.strftime('%Y-%m-%d')}",
            "",
            "## Description",
            "",
            self.description or "_No description provided._",
            "",
        ]
        
        if self.intended_use:
            lines.extend([
                "## Intended Use",
                "",
                f"**Primary Use:** {self.intended_use.primary_use}",
                "",
                f"**Target Population:** {self.intended_use.target_population}",
                "",
                f"**Clinical Settings:** {', '.join(s.value for s in self.intended_use.clinical_setting)}",
                "",
                f"**User Types:** {', '.join(self.intended_use.user_types)}",
                "",
                f"**Regulatory Status:** {self.intended_use.regulatory_status.value}",
                "",
                "### Out of Scope Uses",
                "",
            ])
            for use in self.intended_use.out_of_scope_uses:
                lines.append(f"- {use}")
            lines.append("")
        
        if self.performance:
            lines.extend([
                "## Performance",
                "",
                "### Overall Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Accuracy | {self.performance.overall_accuracy:.4f} |",
                f"| Precision | {self.performance.overall_precision:.4f} |",
                f"| Recall | {self.performance.overall_recall:.4f} |",
                f"| F1 Score | {self.performance.overall_f1:.4f} |",
            ])
            if self.performance.overall_auc_roc:
                lines.append(f"| AUC-ROC | {self.performance.overall_auc_roc:.4f} |")
            lines.append("")
            
            if self.performance.sensitivity and self.performance.specificity:
                lines.extend([
                    "### Clinical Metrics",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Sensitivity | {self.performance.sensitivity:.4f} |",
                    f"| Specificity | {self.performance.specificity:.4f} |",
                ])
                if self.performance.ppv:
                    lines.append(f"| PPV | {self.performance.ppv:.4f} |")
                if self.performance.npv:
                    lines.append(f"| NPV | {self.performance.npv:.4f} |")
                lines.append("")
        
        if self.limitations:
            lines.extend([
                "## Limitations",
                "",
            ])
            for limitation in self.limitations:
                lines.append(f"- {limitation}")
            lines.append("")
        
        if self.ethical_considerations:
            lines.extend([
                "## Ethical Considerations",
                "",
                "### Potential Biases",
                "",
            ])
            for bias in self.ethical_considerations.potential_biases:
                lines.append(f"- {bias}")
            lines.extend([
                "",
                "### Mitigation Strategies",
                "",
            ])
            for strategy in self.ethical_considerations.mitigation_strategies:
                lines.append(f"- {strategy}")
            lines.extend([
                "",
                f"### Human Oversight",
                "",
            ])
            for req in self.ethical_considerations.human_oversight_requirements:
                lines.append(f"- {req}")
            lines.append("")
        
        lines.extend([
            "## Ownership",
            "",
            f"**Organization:** {self.organization or 'Not specified'}  ",
            f"**Contact:** {self.contact or 'Not specified'}  ",
            f"**License:** {self.license or 'Not specified'}",
            "",
            "---",
            f"*Card ID: {self.card_id}*",
        ])
        
        return "\n".join(lines)


class ModelCardGenerator:
    """
    Factory for creating Model Cards.
    
    Supports:
    - Template-based generation
    - Validation and completeness checks
    - Export to multiple formats
    """
    
    def __init__(self):
        self._cards: dict[str, ModelCard] = {}
    
    def create_card(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        model_type: ModelType,
        **kwargs,
    ) -> ModelCard:
        """Create a new model card."""
        card = ModelCard(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            **kwargs,
        )
        
        self._cards[card.card_id] = card
        
        logger.info(
            "model_card_created",
            card_id=card.card_id,
            model_id=model_id,
            model_version=model_version,
        )
        
        return card
    
    def create_healthcare_template(
        self,
        model_id: str,
        model_name: str,
        model_version: str,
        model_type: ModelType,
        risk_level: RiskLevel,
        clinical_settings: list[ClinicalSetting],
        primary_use: str,
        target_population: str,
    ) -> ModelCard:
        """Create a model card with healthcare-specific defaults."""
        
        intended_use = IntendedUse(
            primary_use=primary_use,
            target_population=target_population,
            clinical_setting=clinical_settings,
            user_types=["Licensed healthcare providers"],
            out_of_scope_uses=[
                "Diagnostic decisions without clinical oversight",
                "Use on populations not represented in training data",
                "Emergency triage without human verification",
            ],
            regulatory_status=risk_level,
        )
        
        ethical_considerations = EthicalConsiderations(
            potential_biases=[
                "Model performance may vary across demographic groups",
                "Training data may not represent all patient populations",
            ],
            mitigation_strategies=[
                "Subgroup performance analysis conducted",
                "Regular bias audits scheduled",
                "Human-in-the-loop validation required",
            ],
            fairness_evaluations_performed=[
                "Demographic parity analysis",
                "Equalized odds evaluation",
                "Disparate impact assessment",
            ],
            demographic_groups_analyzed=[
                "Age groups", "Sex/Gender", "Race/Ethnicity", "Geographic region"
            ],
            disparate_impact_assessment="See fairness evaluation report",
            human_oversight_requirements=[
                "All predictions require clinician review",
                "Model outputs are decision support, not autonomous decisions",
                "Regular clinical validation required",
            ],
            explainability_approach="Feature importance and SHAP values available",
        )
        
        card = self.create_card(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            intended_use=intended_use,
            ethical_considerations=ethical_considerations,
            limitations=[
                "Performance validated only on retrospective data",
                "May not generalize to populations outside training distribution",
                "Requires minimum data quality standards",
            ],
            caveats=[
                "Not a substitute for clinical judgment",
                "Performance may degrade with distribution shift",
            ],
        )
        
        return card
    
    def validate_card(self, card: ModelCard) -> tuple[bool, list[str]]:
        """Validate model card completeness for FDA submission."""
        issues = []
        
        # Required fields
        if not card.description:
            issues.append("Missing model description")
        
        if not card.intended_use:
            issues.append("Missing intended use specification")
        elif not card.intended_use.primary_use:
            issues.append("Missing primary use description")
        
        if not card.performance:
            issues.append("Missing performance metrics")
        
        if not card.ethical_considerations:
            issues.append("Missing ethical considerations")
        
        if not card.limitations:
            issues.append("No limitations documented")
        
        if not card.developers:
            issues.append("No developers specified")
        
        if not card.organization:
            issues.append("No organization specified")
        
        is_valid = len(issues) == 0
        
        logger.info(
            "model_card_validation",
            card_id=card.card_id,
            is_valid=is_valid,
            issue_count=len(issues),
        )
        
        return is_valid, issues
    
    def get_card(self, card_id: str) -> ModelCard | None:
        """Retrieve a model card by ID."""
        return self._cards.get(card_id)
    
    def export_card(
        self,
        card: ModelCard,
        format: str = "json",
    ) -> str:
        """Export model card in specified format."""
        if format == "json":
            return card.to_json()
        elif format == "markdown":
            return card.to_markdown()
        else:
            raise ValueError(f"Unsupported format: {format}")
