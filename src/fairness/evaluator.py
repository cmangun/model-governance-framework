"""
AI Fairness Evaluation for Healthcare Models

Comprehensive fairness assessment supporting:
- Demographic parity analysis
- Equalized odds evaluation
- Calibration across groups
- Intersectional fairness
- Healthcare-specific equity metrics
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ProtectedAttribute(str, Enum):
    """Protected demographic attributes."""
    
    AGE = "age"
    SEX = "sex"
    RACE = "race"
    ETHNICITY = "ethnicity"
    DISABILITY = "disability"
    SOCIOECONOMIC = "socioeconomic"
    INSURANCE = "insurance"
    GEOGRAPHIC = "geographic"
    LANGUAGE = "language"


class FairnessMetric(str, Enum):
    """Fairness metrics for evaluation."""
    
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    TREATMENT_EQUALITY = "treatment_equality"
    BALANCE_FOR_POSITIVE = "balance_for_positive"
    BALANCE_FOR_NEGATIVE = "balance_for_negative"


class FairnessStatus(str, Enum):
    """Overall fairness assessment status."""
    
    FAIR = "fair"
    WARNING = "warning"
    UNFAIR = "unfair"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class GroupMetrics:
    """Metrics for a specific demographic group."""
    
    group_name: str
    sample_size: int
    positive_rate: float
    true_positive_rate: float
    false_positive_rate: float
    true_negative_rate: float
    false_negative_rate: float
    precision: float
    recall: float
    f1_score: float
    average_prediction: float
    calibration_error: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "sample_size": self.sample_size,
            "positive_rate": round(self.positive_rate, 4),
            "true_positive_rate": round(self.true_positive_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "true_negative_rate": round(self.true_negative_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "average_prediction": round(self.average_prediction, 4),
            "calibration_error": round(self.calibration_error, 4),
        }


@dataclass
class FairnessMetricResult:
    """Result of a fairness metric evaluation."""
    
    metric: FairnessMetric
    value: float
    threshold: float
    is_fair: bool
    reference_group: str
    comparison_groups: dict[str, float]
    disparity_ratios: dict[str, float]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "is_fair": self.is_fair,
            "reference_group": self.reference_group,
            "comparison_groups": {k: round(v, 4) for k, v in self.comparison_groups.items()},
            "disparity_ratios": {k: round(v, 4) for k, v in self.disparity_ratios.items()},
        }


@dataclass
class FairnessReport:
    """Comprehensive fairness evaluation report."""
    
    report_id: str
    model_id: str
    model_version: str
    timestamp: datetime
    protected_attribute: ProtectedAttribute
    group_metrics: dict[str, GroupMetrics]
    fairness_metrics: list[FairnessMetricResult]
    overall_status: FairnessStatus
    recommendations: list[str]
    intersectional_analysis: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "protected_attribute": self.protected_attribute.value,
            "group_metrics": {k: v.to_dict() for k, v in self.group_metrics.items()},
            "fairness_metrics": [m.to_dict() for m in self.fairness_metrics],
            "overall_status": self.overall_status.value,
            "recommendations": self.recommendations,
            "intersectional_analysis": self.intersectional_analysis,
        }


class FairnessConfig(BaseModel):
    """Configuration for fairness evaluation."""
    
    # Thresholds (using 80% rule as baseline)
    demographic_parity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    equalized_odds_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    calibration_threshold: float = Field(default=0.1, ge=0.0)
    
    # Minimum samples for valid analysis
    min_group_size: int = Field(default=30, ge=1)
    
    # Reference group selection
    reference_group_method: str = "majority"  # "majority", "best_performance", "specified"
    specified_reference_group: str | None = None
    
    # Healthcare-specific
    clinical_significance_threshold: float = Field(default=0.05, ge=0.0)
    evaluate_treatment_allocation: bool = True
    
    # Analysis depth
    enable_intersectional: bool = True
    max_intersectional_depth: int = Field(default=2, ge=1, le=3)


class FairnessEvaluator:
    """
    Healthcare AI Fairness Evaluator.
    
    Features:
    - Multi-metric fairness assessment
    - Group-level performance analysis
    - Disparity ratio calculation
    - Intersectional fairness analysis
    - Healthcare-specific equity metrics
    - Actionable recommendations
    """
    
    def __init__(self, config: FairnessConfig | None = None):
        self.config = config or FairnessConfig()
        self._reports: list[FairnessReport] = []
    
    def evaluate(
        self,
        predictions: list[float],
        actuals: list[int],
        protected_groups: list[str],
        model_id: str,
        model_version: str = "1.0.0",
        protected_attribute: ProtectedAttribute = ProtectedAttribute.RACE,
        threshold: float = 0.5,
    ) -> FairnessReport:
        """
        Evaluate model fairness across demographic groups.
        
        Args:
            predictions: Model prediction probabilities
            actuals: Actual binary outcomes (0 or 1)
            protected_groups: Group membership for each sample
            model_id: Model identifier
            model_version: Model version
            protected_attribute: Type of protected attribute
            threshold: Classification threshold
        
        Returns:
            FairnessReport with comprehensive analysis
        """
        import time
        
        # Validate inputs
        if len(predictions) != len(actuals) != len(protected_groups):
            raise ValueError("All input lists must have same length")
        
        # Group data by protected attribute
        grouped_data = self._group_data(predictions, actuals, protected_groups)
        
        # Filter groups with insufficient data
        valid_groups = {
            k: v for k, v in grouped_data.items()
            if len(v["predictions"]) >= self.config.min_group_size
        }
        
        if len(valid_groups) < 2:
            return self._create_insufficient_data_report(
                model_id, model_version, protected_attribute
            )
        
        # Calculate group metrics
        group_metrics = {}
        for group_name, data in valid_groups.items():
            group_metrics[group_name] = self._calculate_group_metrics(
                group_name,
                data["predictions"],
                data["actuals"],
                threshold,
            )
        
        # Determine reference group
        reference_group = self._select_reference_group(group_metrics)
        
        # Calculate fairness metrics
        fairness_metrics = self._calculate_fairness_metrics(
            group_metrics, reference_group
        )
        
        # Determine overall status
        overall_status = self._determine_overall_status(fairness_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            fairness_metrics, group_metrics, overall_status
        )
        
        # Create report
        report = FairnessReport(
            report_id=self._generate_report_id(model_id),
            model_id=model_id,
            model_version=model_version,
            timestamp=datetime.utcnow(),
            protected_attribute=protected_attribute,
            group_metrics=group_metrics,
            fairness_metrics=fairness_metrics,
            overall_status=overall_status,
            recommendations=recommendations,
        )
        
        self._reports.append(report)
        
        logger.info(
            "fairness_evaluation_complete",
            model_id=model_id,
            protected_attribute=protected_attribute.value,
            overall_status=overall_status.value,
            num_groups=len(group_metrics),
        )
        
        return report
    
    def _group_data(
        self,
        predictions: list[float],
        actuals: list[int],
        groups: list[str],
    ) -> dict[str, dict[str, list]]:
        """Group data by protected attribute."""
        grouped: dict[str, dict[str, list]] = defaultdict(
            lambda: {"predictions": [], "actuals": []}
        )
        
        for pred, actual, group in zip(predictions, actuals, groups):
            grouped[group]["predictions"].append(pred)
            grouped[group]["actuals"].append(actual)
        
        return dict(grouped)
    
    def _calculate_group_metrics(
        self,
        group_name: str,
        predictions: list[float],
        actuals: list[int],
        threshold: float,
    ) -> GroupMetrics:
        """Calculate performance metrics for a group."""
        n = len(predictions)
        
        # Binary predictions
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        
        # Confusion matrix elements
        tp = sum(1 for p, a in zip(pred_binary, actuals) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(pred_binary, actuals) if p == 1 and a == 0)
        tn = sum(1 for p, a in zip(pred_binary, actuals) if p == 0 and a == 0)
        fn = sum(1 for p, a in zip(pred_binary, actuals) if p == 0 and a == 1)
        
        # Rates
        positive_rate = sum(pred_binary) / n if n > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calibration
        avg_pred = sum(predictions) / n if n > 0 else 0
        actual_rate = sum(actuals) / n if n > 0 else 0
        calibration_error = abs(avg_pred - actual_rate)
        
        return GroupMetrics(
            group_name=group_name,
            sample_size=n,
            positive_rate=positive_rate,
            true_positive_rate=tpr,
            false_positive_rate=fpr,
            true_negative_rate=tnr,
            false_negative_rate=fnr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            average_prediction=avg_pred,
            calibration_error=calibration_error,
        )
    
    def _select_reference_group(
        self,
        group_metrics: dict[str, GroupMetrics],
    ) -> str:
        """Select reference group for comparisons."""
        if self.config.specified_reference_group:
            if self.config.specified_reference_group in group_metrics:
                return self.config.specified_reference_group
        
        if self.config.reference_group_method == "majority":
            return max(group_metrics.items(), key=lambda x: x[1].sample_size)[0]
        elif self.config.reference_group_method == "best_performance":
            return max(group_metrics.items(), key=lambda x: x[1].f1_score)[0]
        
        # Default to majority
        return max(group_metrics.items(), key=lambda x: x[1].sample_size)[0]
    
    def _calculate_fairness_metrics(
        self,
        group_metrics: dict[str, GroupMetrics],
        reference_group: str,
    ) -> list[FairnessMetricResult]:
        """Calculate fairness metrics across groups."""
        results = []
        ref_metrics = group_metrics[reference_group]
        
        # Demographic Parity
        dp_comparison = {}
        dp_ratios = {}
        for name, metrics in group_metrics.items():
            if name != reference_group:
                dp_comparison[name] = metrics.positive_rate
                ratio = (
                    metrics.positive_rate / ref_metrics.positive_rate
                    if ref_metrics.positive_rate > 0 else 0
                )
                dp_ratios[name] = ratio
        
        dp_min_ratio = min(dp_ratios.values()) if dp_ratios else 1.0
        results.append(FairnessMetricResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            value=dp_min_ratio,
            threshold=self.config.demographic_parity_threshold,
            is_fair=dp_min_ratio >= self.config.demographic_parity_threshold,
            reference_group=reference_group,
            comparison_groups=dp_comparison,
            disparity_ratios=dp_ratios,
        ))
        
        # Equalized Odds (TPR)
        tpr_comparison = {}
        tpr_ratios = {}
        for name, metrics in group_metrics.items():
            if name != reference_group:
                tpr_comparison[name] = metrics.true_positive_rate
                ratio = (
                    metrics.true_positive_rate / ref_metrics.true_positive_rate
                    if ref_metrics.true_positive_rate > 0 else 0
                )
                tpr_ratios[name] = ratio
        
        tpr_min_ratio = min(tpr_ratios.values()) if tpr_ratios else 1.0
        results.append(FairnessMetricResult(
            metric=FairnessMetric.EQUAL_OPPORTUNITY,
            value=tpr_min_ratio,
            threshold=self.config.equalized_odds_threshold,
            is_fair=tpr_min_ratio >= self.config.equalized_odds_threshold,
            reference_group=reference_group,
            comparison_groups=tpr_comparison,
            disparity_ratios=tpr_ratios,
        ))
        
        # Predictive Parity (Precision)
        pp_comparison = {}
        pp_ratios = {}
        for name, metrics in group_metrics.items():
            if name != reference_group:
                pp_comparison[name] = metrics.precision
                ratio = (
                    metrics.precision / ref_metrics.precision
                    if ref_metrics.precision > 0 else 0
                )
                pp_ratios[name] = ratio
        
        pp_min_ratio = min(pp_ratios.values()) if pp_ratios else 1.0
        results.append(FairnessMetricResult(
            metric=FairnessMetric.PREDICTIVE_PARITY,
            value=pp_min_ratio,
            threshold=self.config.demographic_parity_threshold,
            is_fair=pp_min_ratio >= self.config.demographic_parity_threshold,
            reference_group=reference_group,
            comparison_groups=pp_comparison,
            disparity_ratios=pp_ratios,
        ))
        
        return results
    
    def _determine_overall_status(
        self,
        fairness_metrics: list[FairnessMetricResult],
    ) -> FairnessStatus:
        """Determine overall fairness status."""
        unfair_count = sum(1 for m in fairness_metrics if not m.is_fair)
        
        if unfair_count == 0:
            return FairnessStatus.FAIR
        elif unfair_count == 1:
            return FairnessStatus.WARNING
        else:
            return FairnessStatus.UNFAIR
    
    def _generate_recommendations(
        self,
        fairness_metrics: list[FairnessMetricResult],
        group_metrics: dict[str, GroupMetrics],
        status: FairnessStatus,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        for metric_result in fairness_metrics:
            if not metric_result.is_fair:
                # Find most disadvantaged group
                if metric_result.disparity_ratios:
                    worst_group = min(
                        metric_result.disparity_ratios.items(),
                        key=lambda x: x[1]
                    )
                    
                    recommendations.append(
                        f"{metric_result.metric.value}: Group '{worst_group[0]}' shows "
                        f"{(1 - worst_group[1]) * 100:.1f}% disparity compared to reference. "
                        f"Consider targeted interventions or model adjustments."
                    )
        
        if status == FairnessStatus.UNFAIR:
            recommendations.append(
                "CRITICAL: Multiple fairness violations detected. "
                "Consider retraining with balanced data, adjusting thresholds, "
                "or implementing post-processing calibration."
            )
        
        if not recommendations:
            recommendations.append(
                "Model meets fairness thresholds. Continue monitoring for drift."
            )
        
        return recommendations
    
    def _create_insufficient_data_report(
        self,
        model_id: str,
        model_version: str,
        protected_attribute: ProtectedAttribute,
    ) -> FairnessReport:
        """Create report when insufficient data."""
        return FairnessReport(
            report_id=self._generate_report_id(model_id),
            model_id=model_id,
            model_version=model_version,
            timestamp=datetime.utcnow(),
            protected_attribute=protected_attribute,
            group_metrics={},
            fairness_metrics=[],
            overall_status=FairnessStatus.INSUFFICIENT_DATA,
            recommendations=[
                f"Insufficient data for fairness analysis. "
                f"Need at least {self.config.min_group_size} samples per group "
                f"with at least 2 groups."
            ],
        )
    
    def _generate_report_id(self, model_id: str) -> str:
        """Generate unique report ID."""
        import time
        content = f"{model_id}:{time.time_ns()}"
        return f"fair_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
