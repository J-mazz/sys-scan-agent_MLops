"""
Base verifier class for validating synthetic data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import json

class BaseVerifier(ABC):
    """Base class for all synthetic data verifiers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def verify(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify the synthetic data.

        Args:
            data: The synthetic data to verify

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass

    def _log_issue(self, issue: str) -> str:
        """Format an issue with verifier name."""
        return f"[{self.name}] {issue}"

    # ----------------------
    # Shared validation helpers
    # ----------------------
    @property
    def _valid_severities(self) -> List[str]:
        return ["info", "low", "medium", "high", "critical"]

    def _validate_required_fields(
        self,
        obj: Dict[str, Any],
        required_fields: List[str],
        prefix: str
    ) -> List[str]:
        issues = []
        for field in required_fields:
            if field not in obj:
                issues.append(self._log_issue(f"{prefix}: Missing required field '{field}'"))
        return issues

    def _validate_severity_and_scores(
        self,
        item: Dict[str, Any],
        prefix: str,
        score_fields: Optional[List[str]] = None,
    ) -> List[str]:
        issues = []
        if "severity" in item and item["severity"] not in self._valid_severities:
            issues.append(self._log_issue(f"{prefix}: Invalid severity '{item['severity']}'"))

        for score_field in score_fields or ["risk_score", "base_severity_score"]:
            if score_field in item:
                value = item[score_field]
                if not isinstance(value, (int, float)):
                    issues.append(self._log_issue(f"{prefix}: {score_field} must be numeric"))
                elif value < 0 or value > 100:
                    issues.append(self._log_issue(f"{prefix}: {score_field} out of range 0-100 (got {value})"))
        return issues

    def _validate_probability(
        self,
        item: Dict[str, Any],
        field: str,
        prefix: str
    ) -> List[str]:
        issues = []
        if field in item:
            value = item[field]
            if not isinstance(value, (int, float)):
                issues.append(self._log_issue(f"{prefix}: {field} must be numeric"))
            elif value < 0 or value > 1:
                issues.append(self._log_issue(f"{prefix}: {field} must be between 0 and 1 (got {value})"))
        return issues

    def _validate_risk_subscores(self, subscores: Any, prefix: str) -> List[str]:
        issues = []
        required_subscores = ["impact", "exposure", "anomaly", "confidence"]
        if not isinstance(subscores, dict):
            issues.append(self._log_issue(f"{prefix}: risk_subscores must be a dict"))
            return issues
        for subscore in required_subscores:
            if subscore not in subscores:
                issues.append(self._log_issue(f"{prefix}: Missing risk_subscore '{subscore}'"))
            else:
                value = subscores.get(subscore)
                if not isinstance(value, (int, float)):
                    issues.append(self._log_issue(f"{prefix}: risk_subscore '{subscore}' must be numeric"))
                elif value < 0 or value > 1:
                    issues.append(self._log_issue(f"{prefix}: risk_subscore '{subscore}' out of range 0-1 (got {value})"))
        return issues

    def _validate_host_metadata(self, metadata: Dict[str, Any], prefix: str) -> List[str]:
        issues = []
        host_keys = ["distro", "distro_version", "package_manager", "kernel"]
        if not isinstance(metadata, dict):
            issues.append(self._log_issue(f"{prefix}: metadata must be a dict"))
            return issues
        for key in host_keys:
            if metadata.get(key) in (None, ""):
                issues.append(self._log_issue(f"{prefix}: metadata missing '{key}'"))
        kb_refs = metadata.get("kb_refs")
        if kb_refs is None or not isinstance(kb_refs, list) or len(kb_refs) == 0:
            issues.append(self._log_issue(f"{prefix}: metadata.kb_refs must be a non-empty list"))
        return issues