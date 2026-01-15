"""Tests for Model Cards Generator."""
import pytest
from datetime import datetime
from src.model_cards.generator import (
    ModelCard,
    ModelCardGenerator,
    ModelType,
    RiskLevel,
    ClinicalSetting,
    IntendedUse,
    PerformanceMetrics,
    EthicalConsiderations,
)


class TestModelCard:
    """Tests for ModelCard dataclass."""
    
    def test_create_basic_card(self):
        """Test creating a basic model card."""
        card = ModelCard(
            model_id="test-001",
            model_name="Test Model",
            model_version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
        )
        
        assert card.model_id == "test-001"
        assert card.model_name == "Test Model"
        assert card.model_version == "1.0.0"
        assert card.model_type == ModelType.CLASSIFICATION
        assert card.card_id.startswith("mc_")
    
    def test_card_to_dict(self):
        """Test converting card to dictionary."""
        card = ModelCard(
            model_id="test-001",
            model_name="Test Model",
            model_version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            description="A test model for unit testing",
        )
        
        result = card.to_dict()
        
        assert result["model_identity"]["model_id"] == "test-001"
        assert result["description"] == "A test model for unit testing"
        assert "timestamps" in result
    
    def test_card_to_json(self):
        """Test JSON serialization."""
        card = ModelCard(
            model_id="test-001",
            model_name="Test Model",
            model_version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
        )
        
        json_str = card.to_json()
        
        assert '"model_id": "test-001"' in json_str
        assert '"model_name": "Test Model"' in json_str
    
    def test_card_to_markdown(self):
        """Test markdown generation."""
        card = ModelCard(
            model_id="test-001",
            model_name="Test Model",
            model_version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            description="A test model",
        )
        
        markdown = card.to_markdown()
        
        assert "# Model Card: Test Model" in markdown
        assert "**Model ID:** test-001" in markdown
        assert "A test model" in markdown


class TestIntendedUse:
    """Tests for IntendedUse."""
    
    def test_intended_use_creation(self):
        """Test creating intended use specification."""
        intended_use = IntendedUse(
            primary_use="Risk prediction",
            target_population="Adult patients",
            clinical_setting=[ClinicalSetting.INPATIENT],
            user_types=["Clinicians"],
            out_of_scope_uses=["Pediatric patients"],
            regulatory_status=RiskLevel.CLASS_II,
        )
        
        assert intended_use.primary_use == "Risk prediction"
        assert ClinicalSetting.INPATIENT in intended_use.clinical_setting
    
    def test_intended_use_to_dict(self):
        """Test intended use serialization."""
        intended_use = IntendedUse(
            primary_use="Diagnosis support",
            target_population="Adults 18+",
            clinical_setting=[ClinicalSetting.PRIMARY_CARE, ClinicalSetting.SPECIALTY],
            user_types=["Physicians", "NPs"],
            out_of_scope_uses=["Emergency triage"],
            regulatory_status=RiskLevel.CLASS_II,
        )
        
        result = intended_use.to_dict()
        
        assert result["primary_use"] == "Diagnosis support"
        assert "primary_care" in result["clinical_setting"]
        assert result["regulatory_status"] == "class_ii"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            overall_accuracy=0.92,
            overall_precision=0.89,
            overall_recall=0.94,
            overall_f1=0.91,
            overall_auc_roc=0.96,
            sensitivity=0.94,
            specificity=0.90,
        )
        
        assert metrics.overall_accuracy == 0.92
        assert metrics.sensitivity == 0.94
    
    def test_performance_metrics_to_dict(self):
        """Test performance metrics serialization."""
        metrics = PerformanceMetrics(
            overall_accuracy=0.92,
            overall_precision=0.89,
            overall_recall=0.94,
            overall_f1=0.91,
            eval_dataset_size=10000,
        )
        
        result = metrics.to_dict()
        
        assert result["overall"]["accuracy"] == 0.92
        assert result["evaluation_info"]["dataset_size"] == 10000


class TestModelCardGenerator:
    """Tests for ModelCardGenerator."""
    
    def test_create_card(self):
        """Test creating a card via generator."""
        generator = ModelCardGenerator()
        
        card = generator.create_card(
            model_id="gen-001",
            model_name="Generated Model",
            model_version="2.0.0",
            model_type=ModelType.NLP,
        )
        
        assert card.model_id == "gen-001"
        assert card.model_type == ModelType.NLP
    
    def test_create_healthcare_template(self):
        """Test creating healthcare-specific card."""
        generator = ModelCardGenerator()
        
        card = generator.create_healthcare_template(
            model_id="health-001",
            model_name="Sepsis Predictor",
            model_version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            risk_level=RiskLevel.CLASS_II,
            clinical_settings=[ClinicalSetting.INPATIENT, ClinicalSetting.EMERGENCY],
            primary_use="Early sepsis detection",
            target_population="Adult ICU patients",
        )
        
        assert card.intended_use is not None
        assert card.ethical_considerations is not None
        assert len(card.limitations) > 0
        assert card.intended_use.regulatory_status == RiskLevel.CLASS_II
    
    def test_validate_incomplete_card(self):
        """Test validation catches missing fields."""
        generator = ModelCardGenerator()
        
        card = ModelCard(
            model_id="incomplete",
            model_name="Incomplete Model",
            model_version="0.1.0",
            model_type=ModelType.CLASSIFICATION,
        )
        
        is_valid, issues = generator.validate_card(card)
        
        assert not is_valid
        assert len(issues) > 0
        assert "Missing model description" in issues
    
    def test_validate_complete_card(self):
        """Test validation passes for complete card."""
        generator = ModelCardGenerator()
        
        card = generator.create_healthcare_template(
            model_id="complete-001",
            model_name="Complete Model",
            model_version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            risk_level=RiskLevel.CLASS_II,
            clinical_settings=[ClinicalSetting.INPATIENT],
            primary_use="Risk prediction",
            target_population="Adult patients",
        )
        
        # Add remaining required fields
        card.description = "A complete model for testing"
        card.performance = PerformanceMetrics(
            overall_accuracy=0.90,
            overall_precision=0.88,
            overall_recall=0.92,
            overall_f1=0.90,
        )
        card.developers = ["Test Developer"]
        card.organization = "Test Org"
        
        is_valid, issues = generator.validate_card(card)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_export_json(self):
        """Test JSON export."""
        generator = ModelCardGenerator()
        
        card = generator.create_card(
            model_id="export-001",
            model_name="Export Test",
            model_version="1.0.0",
            model_type=ModelType.DETECTION,
        )
        
        json_output = generator.export_card(card, format="json")
        
        assert '"model_id": "export-001"' in json_output
    
    def test_export_markdown(self):
        """Test markdown export."""
        generator = ModelCardGenerator()
        
        card = generator.create_card(
            model_id="export-002",
            model_name="Markdown Export",
            model_version="1.0.0",
            model_type=ModelType.SEGMENTATION,
            description="Test markdown export",
        )
        
        md_output = generator.export_card(card, format="markdown")
        
        assert "# Model Card: Markdown Export" in md_output
        assert "Test markdown export" in md_output


class TestEthicalConsiderations:
    """Tests for EthicalConsiderations."""
    
    def test_ethical_considerations_creation(self):
        """Test creating ethical considerations."""
        ethics = EthicalConsiderations(
            potential_biases=["Age bias", "Geographic bias"],
            mitigation_strategies=["Balanced sampling", "Regular audits"],
            fairness_evaluations_performed=["Demographic parity"],
            demographic_groups_analyzed=["Age", "Gender"],
            disparate_impact_assessment="Passed 80% rule",
            human_oversight_requirements=["Clinician review required"],
            explainability_approach="SHAP values",
        )
        
        assert len(ethics.potential_biases) == 2
        assert "SHAP values" in ethics.explainability_approach
    
    def test_ethical_considerations_to_dict(self):
        """Test ethical considerations serialization."""
        ethics = EthicalConsiderations(
            potential_biases=["Selection bias"],
            mitigation_strategies=["Resampling"],
            fairness_evaluations_performed=["Equalized odds"],
            demographic_groups_analyzed=["Race"],
            disparate_impact_assessment="Under review",
            human_oversight_requirements=["Required"],
            explainability_approach="Feature importance",
        )
        
        result = ethics.to_dict()
        
        assert result["potential_biases"] == ["Selection bias"]
        assert result["explainability_approach"] == "Feature importance"
