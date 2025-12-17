"""
Schema verifier to ensure synthetic data matches the ground truth schema.
"""

from typing import Dict, List, Any, Tuple
import json
import os
from .base_verifier import BaseVerifier

class SchemaVerifier(BaseVerifier):
    """Verifier for JSON schema compliance."""

    def __init__(self):
        super().__init__("SchemaVerifier")
        self.schema_path = os.path.join(os.path.dirname(__file__), "ground_truth_schema.json")
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Load the ground truth schema."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def verify(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify data against the schema."""
        issues = []

        # Check required top-level fields
        required_fields = ["version", "enriched_findings", "correlations", "reductions", "summaries", "actions"]
        for field in required_fields:
            if field not in data:
                issues.append(self._log_issue(f"Missing required field: {field}"))

        # Check version
        if "version" in data and data["version"] != "ground_truth_v1":
            issues.append(self._log_issue(f"Invalid version: {data.get('version')}"))

        # Check enriched_findings structure
        if "enriched_findings" in data:
            findings = data["enriched_findings"]
            if not isinstance(findings, list):
                issues.append(self._log_issue("enriched_findings must be a list"))
            else:
                for i, finding in enumerate(findings):
                    finding_issues = self._verify_finding_structure(finding, i)
                    issues.extend(finding_issues)

        # Check correlations structure
        if "correlations" in data:
            correlations = data["correlations"]
            if not isinstance(correlations, list):
                issues.append(self._log_issue("correlations must be a list"))
            else:
                for i, correlation in enumerate(correlations):
                    correlation_issues = self._verify_correlation_structure(correlation, i)
                    issues.extend(correlation_issues)

        return len(issues) == 0, issues

    def _verify_finding_structure(self, finding: Dict[str, Any], index: int) -> List[str]:
        """Verify a single finding's structure."""
        issues: List[str] = []
        required_fields = [
            "id",
            "title",
            "severity",
            "risk_score",
            "base_severity_score",
            "description",
            "metadata",
            "risk_subscores",
            "probability_actionable",
            "baseline_status",
            "tags",
            "rationale",
            "correlation_refs",
        ]

        issues.extend(self._validate_required_fields(finding, required_fields, f"Finding {index}"))
        issues.extend(self._validate_severity_and_scores(finding, f"Finding {index}"))
        issues.extend(self._validate_probability(finding, "probability_actionable", f"Finding {index}"))

        if "risk_subscores" in finding:
            issues.extend(self._validate_risk_subscores(finding.get("risk_subscores"), f"Finding {index}"))

        metadata = finding.get("metadata", {})
        issues.extend(self._validate_host_metadata(metadata, f"Finding {index}"))

        return issues

    def _verify_correlation_structure(self, correlation: Dict[str, Any], index: int) -> List[str]:
        """Verify a single correlation's structure."""
        issues: List[str] = []
        required_fields = [
            "id",
            "title",
            "severity",
            "risk_score",
            "base_severity_score",
            "description",
            "metadata",
            "risk_subscores",
            "probability_actionable",
            "baseline_status",
            "tags",
            "rationale",
            "correlation_refs",
            "correlation_type",
        ]

        issues.extend(self._validate_required_fields(correlation, required_fields, f"Correlation {index}"))
        issues.extend(self._validate_severity_and_scores(correlation, f"Correlation {index}"))
        issues.extend(self._validate_probability(correlation, "probability_actionable", f"Correlation {index}"))

        if "risk_subscores" in correlation:
            issues.extend(self._validate_risk_subscores(correlation.get("risk_subscores"), f"Correlation {index}"))

        metadata = correlation.get("metadata", {})
        issues.extend(self._validate_host_metadata(metadata, f"Correlation {index}"))

        # Validate correlation_refs list
        if "correlation_refs" in correlation and not isinstance(correlation["correlation_refs"], list):
            issues.append(self._log_issue(f"Correlation {index}: correlation_refs must be a list"))
        elif "correlation_refs" in correlation and len(correlation.get("correlation_refs") or []) == 0:
            issues.append(self._log_issue(f"Correlation {index}: correlation_refs must not be empty"))

        return issues