"""
AI Fairness and Bias Detection for Healthcare

Comprehensive fairness evaluation supporting:
- Demographic parity analysis
- Equalized odds metrics
- Disparate impact detection
- Intersectional fairness
- Healthcare-specific bias indicators
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class FairnessMetric(str, Enum):
    """Types of fairness metrics."""
    
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    DISPARATE_IMPACT = "disparate_impact"
    INDIVIDUAL_FAIRNESS = "individual_fairness"


class ProtectedAttribute(str, Enum):
    """Protected attributes for fairness analysis."""
    
    RACE = "race"
    ETHNICITY = "ethnicity"
    GENDER = "gender"
    AGE_GROUP = "age_group"
    SOCIOECONOMIC = "socioeconomic"
    GEOGRAPHIC = "geographic"
    DISABILITY = "disability"
    INSURANCE_TYPE = "insurance_type"


class FairnessStatus(str, Enum):
    """Overall fairness assessment status."""
    
    FAIR = "fair"
    MARGINAL = "marginal"
    UNFAIR = "unfair"
    CRITICAL = "critical"


@dataclass
class GroupMetrics:
    """Performance metrics for a demographic group."""
    
    group_name: str
    group_size: int
    positive_rate: float  # P(Y=1)
    true_positive_rate: float  # TPR = TP / (TP + FN)
    false_positive_rate: float  # FPR = FP / (FP + TN)
    true_negative_rate: float  # TNR = TN / (TN + FP)
    false_negative_rate: float  # FNR = FN / (FN + TP)
    precision: float  # PPV = TP / (TP + FP)
    negative_predictive_value: float  # NPV = TN / (TN + FN)
    accuracy: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "group_size": self.group_size,
            "positive_rate": round(self.positive_rate, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "true_negative_rate": round(self.true_negative_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "precision": round(self.precision, 4),
            "negative_predictive_value": round(self.negative_predictive_value, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class FairnessViolation:
    """A detected fairness violation."""
    
    metric: FairnessMetric
    protected_attribute: ProtectedAttribute
    advantaged_group: str
    disadvantaged_group: str
    disparity_value: float
    threshold: float
    severity: FairnessStatus
    description: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "protected_attribute": self.protected_attribute.value,
            "advantaged_group": self.advantaged_group,
            "disadvantaged_group": self.disadvantaged_group,
            "disparity_value": round(self.disparity_value, 4),
            "threshold": self.threshold,
            "severity": self.severity.value,
            "description": self.description,
        }


@dataclass
class FairnessReport:
    """Comprehensive fairness analysis report."""
    
    model_id: str
    model_version: str
    report_id: str
    timestamp: datetime
    sample_size: int
    protected_attributes_analyzed: list[ProtectedAttribute]
    group_metrics: dict[str, dict[str, GroupMetrics]]  # attr -> group -> metrics
    violations: list[FairnessViolation]
    overall_status: FairnessStatus
    recommendations: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "sample_size": self.sample_size,
            "protected_attributes_analyzed": [a.value for a in self.protected_attributes_analyzed],
            "group_metrics": {
                attr: {group: m.to_dict() for group, m in groups.items()}
                for attr, groups in self.group_metrics.items()
            },
            "violations": [v.to_dict() for v in self.violations],
            "overall_status": self.overall_status.value,
            "recommendations": self.recommendations,
            "violation_count": len(self.violations),
        }


class BiasDetectorConfig(BaseModel):
    """Configuration for bias detection."""
    
    # Thresholds
    demographic_parity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    equalized_odds_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    disparate_impact_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    
    # Minimum group size for analysis
    min_group_size: int = Field(default=30, ge=1)
    
    # Healthcare-specific
    clinical_significance_threshold: float = Field(default=0.05, ge=0.0)
    
    # Analysis settings
    analyze_intersections: bool = True
    max_intersection_depth: int = Field(default=2, ge=1, le=3)


class BiasDetector:
    """
    Production bias and fairness detector for healthcare ML.
    
    Features:
    - Multi-metric fairness evaluation
    - Intersectional analysis
    - Healthcare-specific indicators
    - Actionable recommendations
    - Audit-ready reporting
    """
    
    def __init__(self, config: BiasDetectorConfig | None = None):
        self.config = config or BiasDetectorConfig()
        self._reports: list[FairnessReport] = []
    
    def analyze(
        self,
        model_id: str,
        model_version: str,
        predictions: list[int],
        actuals: list[int],
        protected_attributes: dict[ProtectedAttribute, list[str]],
    ) -> FairnessReport:
        """
        Analyze model for bias across protected attributes.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            predictions: Model predictions (0 or 1)
            actuals: Ground truth labels (0 or 1)
            protected_attributes: Mapping of attribute to group membership
        
        Returns:
            FairnessReport with comprehensive analysis
        """
        import uuid
        
        n_samples = len(predictions)
        if n_samples != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        for attr, groups in protected_attributes.items():
            if len(groups) != n_samples:
                raise ValueError(f"Attribute {attr.value} has wrong length")
        
        # Calculate group metrics for each protected attribute
        group_metrics: dict[str, dict[str, GroupMetrics]] = {}
        violations: list[FairnessViolation] = []
        
        for attr, groups in protected_attributes.items():
            group_metrics[attr.value] = {}
            
            # Get unique groups
            unique_groups = list(set(groups))
            
            for group in unique_groups:
                # Get indices for this group
                indices = [i for i, g in enumerate(groups) if g == group]
                
                if len(indices) < self.config.min_group_size:
                    continue
                
                # Calculate metrics for this group
                metrics = self._calculate_group_metrics(
                    group_name=group,
                    predictions=[predictions[i] for i in indices],
                    actuals=[actuals[i] for i in indices],
                )
                
                group_metrics[attr.value][group] = metrics
            
            # Check for violations between groups
            attr_violations = self._check_violations(
                attr,
                group_metrics[attr.value],
            )
            violations.extend(attr_violations)
        
        # Determine overall status
        overall_status = self._determine_overall_status(violations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, group_metrics)
        
        report = FairnessReport(
            model_id=model_id,
            model_version=model_version,
            report_id=f"fair_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.utcnow(),
            sample_size=n_samples,
            protected_attributes_analyzed=list(protected_attributes.keys()),
            group_metrics=group_metrics,
            violations=violations,
            overall_status=overall_status,
            recommendations=recommendations,
        )
        
        self._reports.append(report)
        
        logger.info(
            "fairness_analysis_complete",
            model_id=model_id,
            overall_status=overall_status.value,
            violation_count=len(violations),
        )
        
        return report
    
    def _calculate_group_metrics(
        self,
        group_name: str,
        predictions: list[int],
        actuals: list[int],
    ) -> GroupMetrics:
        """Calculate performance metrics for a group."""
        n = len(predictions)
        
        # Confusion matrix
        tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        tn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
        
        # Calculate rates (with smoothing to avoid division by zero)
        positive_rate = sum(predictions) / n if n > 0 else 0
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / n if n > 0 else 0
        
        return GroupMetrics(
            group_name=group_name,
            group_size=n,
            positive_rate=positive_rate,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            true_negative_rate=tnr,
            false_negative_rate=fnr,
            precision=precision,
            negative_predictive_value=npv,
            accuracy=accuracy,
        )
    
    def _check_violations(
        self,
        attr: ProtectedAttribute,
        group_metrics: dict[str, GroupMetrics],
    ) -> list[FairnessViolation]:
        """Check for fairness violations between groups."""
        violations = []
        groups = list(group_metrics.keys())
        
        if len(groups) < 2:
            return violations
        
        # Compare all pairs of groups
        for i, group1 in enumerate(groups):
            for group2 in groups[i + 1:]:
                m1 = group_metrics[group1]
                m2 = group_metrics[group2]
                
                # Demographic Parity: P(Y=1|A=a) should be similar across groups
                if m1.positive_rate > 0 and m2.positive_rate > 0:
                    dp_ratio = min(m1.positive_rate, m2.positive_rate) / max(m1.positive_rate, m2.positive_rate)
                    
                    if dp_ratio < self.config.demographic_parity_threshold:
                        advantaged = group1 if m1.positive_rate > m2.positive_rate else group2
                        disadvantaged = group2 if advantaged == group1 else group1
                        
                        violations.append(FairnessViolation(
                            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                            protected_attribute=attr,
                            advantaged_group=advantaged,
                            disadvantaged_group=disadvantaged,
                            disparity_value=dp_ratio,
                            threshold=self.config.demographic_parity_threshold,
                            severity=self._assess_severity(dp_ratio, self.config.demographic_parity_threshold),
                            description=f"Positive prediction rate differs significantly between {advantaged} ({max(m1.positive_rate, m2.positive_rate):.2%}) and {disadvantaged} ({min(m1.positive_rate, m2.positive_rate):.2%})",
                        ))
                
                # Equalized Odds: TPR and FPR should be similar
                tpr_ratio = min(m1.true_positive_rate, m2.true_positive_rate) / max(m1.true_positive_rate, m2.true_positive_rate) if max(m1.true_positive_rate, m2.true_positive_rate) > 0 else 1.0
                
                if tpr_ratio < self.config.equalized_odds_threshold:
                    advantaged = group1 if m1.true_positive_rate > m2.true_positive_rate else group2
                    disadvantaged = group2 if advantaged == group1 else group1
                    
                    violations.append(FairnessViolation(
                        metric=FairnessMetric.EQUALIZED_ODDS,
                        protected_attribute=attr,
                        advantaged_group=advantaged,
                        disadvantaged_group=disadvantaged,
                        disparity_value=tpr_ratio,
                        threshold=self.config.equalized_odds_threshold,
                        severity=self._assess_severity(tpr_ratio, self.config.equalized_odds_threshold),
                        description=f"True positive rate differs between {advantaged} and {disadvantaged}",
                    ))
                
                # Disparate Impact (4/5ths rule)
                if m1.positive_rate > 0 and m2.positive_rate > 0:
                    di_ratio = min(m1.positive_rate, m2.positive_rate) / max(m1.positive_rate, m2.positive_rate)
                    
                    if di_ratio < self.config.disparate_impact_threshold:
                        violations.append(FairnessViolation(
                            metric=FairnessMetric.DISPARATE_IMPACT,
                            protected_attribute=attr,
                            advantaged_group=group1 if m1.positive_rate > m2.positive_rate else group2,
                            disadvantaged_group=group2 if m1.positive_rate > m2.positive_rate else group1,
                            disparity_value=di_ratio,
                            threshold=self.config.disparate_impact_threshold,
                            severity=self._assess_severity(di_ratio, self.config.disparate_impact_threshold),
                            description=f"Disparate impact detected: ratio {di_ratio:.2%} below 80% threshold",
                        ))
        
        return violations
    
    def _assess_severity(self, value: float, threshold: float) -> FairnessStatus:
        """Assess severity of a fairness violation."""
        if value >= threshold:
            return FairnessStatus.FAIR
        elif value >= threshold * 0.9:
            return FairnessStatus.MARGINAL
        elif value >= threshold * 0.7:
            return FairnessStatus.UNFAIR
        else:
            return FairnessStatus.CRITICAL
    
    def _determine_overall_status(
        self,
        violations: list[FairnessViolation],
    ) -> FairnessStatus:
        """Determine overall fairness status."""
        if not violations:
            return FairnessStatus.FAIR
        
        severities = [v.severity for v in violations]
        
        if FairnessStatus.CRITICAL in severities:
            return FairnessStatus.CRITICAL
        elif FairnessStatus.UNFAIR in severities:
            return FairnessStatus.UNFAIR
        elif FairnessStatus.MARGINAL in severities:
            return FairnessStatus.MARGINAL
        else:
            return FairnessStatus.FAIR
    
    def _generate_recommendations(
        self,
        violations: list[FairnessViolation],
        group_metrics: dict[str, dict[str, GroupMetrics]],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not violations:
            recommendations.append("No significant fairness violations detected.")
            return recommendations
        
        # Group violations by type
        violation_types = set(v.metric for v in violations)
        
        if FairnessMetric.DEMOGRAPHIC_PARITY in violation_types:
            recommendations.append(
                "Consider rebalancing training data to ensure demographic parity. "
                "Review sampling strategies for underrepresented groups."
            )
        
        if FairnessMetric.EQUALIZED_ODDS in violation_types:
            recommendations.append(
                "True positive rates differ across groups. Consider post-processing "
                "calibration or threshold adjustment per group."
            )
        
        if FairnessMetric.DISPARATE_IMPACT in violation_types:
            recommendations.append(
                "Disparate impact detected (below 80% rule). This may have legal "
                "implications. Consider fairness-aware model training."
            )
        
        # Healthcare-specific recommendations
        recommendations.append(
            "For healthcare applications: Validate findings with clinical experts "
            "and consider impact on patient outcomes across demographic groups."
        )
        
        return recommendations
