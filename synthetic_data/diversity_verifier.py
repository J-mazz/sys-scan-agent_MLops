"""Verifier that ensures dataset diversity across severities and scanner categories."""

from typing import Any, Dict, List, Tuple
from collections import Counter
from base_verifier import BaseVerifier

class DiversityVerifier(BaseVerifier):
    """Validates that generated findings cover a healthy range of signals."""

    def __init__(self) -> None:
        super().__init__("DiversityVerifier")

    def verify(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues: List[str] = []

        findings: List[Dict[str, Any]] = data.get("enriched_findings", [])
        if not findings:
            return False, [self._log_issue("No findings available for diversity assessment")]

        categories = Counter(f.get("category", "unknown") for f in findings)
        severities = Counter(f.get("severity", "unknown") for f in findings)

        if len(categories) < 4:
            issues.append(self._log_issue("Insufficient scanner coverage detected"))

        dominant_category, dominant_count = categories.most_common(1)[0]
        if dominant_count / max(len(findings), 1) > 0.6:
            issues.append(
                self._log_issue(
                    f"Category '{dominant_category}' accounts for more than 60% of findings"
                )
            )

        severity_span = {sev for sev, count in severities.items() if count > 0}
        expected_severities = {"info", "low", "medium", "high", "critical"}
        if len(severity_span & expected_severities) < 4:
            issues.append(self._log_issue("Severity distribution lacks expected coverage"))

        extreme_high = sum(1 for f in findings if f.get("severity") in {"high", "critical"})
        if extreme_high / max(len(findings), 1) > 0.4:
            issues.append(self._log_issue("Excessive high/critical findings detected"))

        return len(issues) == 0, issues
