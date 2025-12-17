"""
Base producer class for synthetic data generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import random
import os
from collections import Counter
from datetime import datetime
import uuid

class BaseProducer(ABC):
    """Base class for all synthetic data producers."""

    def __init__(self, scanner_name: str):
        self.scanner_name = scanner_name
        # Tilt toward more informational/low findings to satisfy realism checks while keeping
        # enough medium/high for signal. Realism expects ~40% info, ~30% low, modest high/critical.
        self.scenarios = {
            'normal': 0.45,      # info-leaning
            'suspicious': 0.35,  # medium
            'malicious': 0.15,   # high/critical
            'edge_case': 0.05,   # rare edge scenarios
        }

    @abstractmethod
    def generate_findings(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic findings for this scanner.

        Args:
            count: Number of findings to generate

        Returns:
            List of finding dictionaries matching the ground truth schema
        """
        pass

    def _generate_base_finding(self, finding_id: str, title: str, severity: str,
                              risk_score: int, base_severity_score: int,
                              description: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a base finding structure."""
        return {
            "id": finding_id,
            "title": title,
            "severity": severity,
            "risk_score": risk_score,
            "base_severity_score": base_severity_score,
            "description": description,
            "metadata": metadata,
            "operational_error": False,
            "category": self.scanner_name,
            "tags": self._generate_tags(severity),
            "risk_subscores": self._generate_risk_subscores(severity),
            "correlation_refs": [],
            "baseline_status": random.choice(["new", "existing", "unknown"]),
            "severity_source": "raw",
            "allowlist_reason": None,
            "probability_actionable": self._calculate_probability_actionable(risk_score),
            "graph_degree": None,
            "cluster_id": None,
            "rationale": None,
            "risk_total": risk_score,
            "host_role": None,
            "host_role_rationale": None,
            "metric_drift": None
        }

    def _generate_tags(self, severity: str) -> List[str]:
        """Generate appropriate tags based on severity."""
        base_tags = [self.scanner_name, f"baseline:{random.choice(['new', 'existing'])}"]

        if severity in ['high', 'critical']:
            base_tags.extend(['high_priority', 'needs_attention'])
        elif severity == 'medium':
            base_tags.append('moderate_risk')
        elif severity in ['low', 'info']:
            base_tags.append('low_priority')

        return base_tags

    def _generate_risk_subscores(self, severity: str) -> Dict[str, float]:
        """Generate risk subscores based on severity."""
        severity_multipliers = {
            'info': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }

        multiplier = severity_multipliers.get(severity, 0.5)

        return {
            "impact": round(random.uniform(0.1, 1.0) * multiplier, 2),
            "exposure": round(random.uniform(0.1, 1.0) * multiplier, 2),
            "anomaly": round(random.uniform(0.1, 1.0) * multiplier, 2),
            "confidence": round(random.uniform(0.7, 0.95), 2)
        }

    def _calculate_probability_actionable(self, risk_score: int) -> float:
        """Calculate probability that finding is actionable."""
        # Higher risk scores are more likely to be actionable
        base_prob = risk_score / 100.0
        return round(min(1.0, base_prob + random.uniform(-0.1, 0.1)), 3)

    def _choose_scenario(self) -> str:
        """Randomly choose a scenario based on weights."""
        rand = random.random()
        cumulative = 0.0
        for scenario, weight in self.scenarios.items():
            cumulative += weight
            if rand <= cumulative:
                return scenario
        return 'normal'


class AggregatingProducer(BaseProducer):
    """Producer that aggregates similar findings into knowledge-dense clusters."""

    def __init__(self, scanner_name: str):
        super().__init__(scanner_name)

    def aggregate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate similar findings into dense summaries."""
        if not findings:
            return findings

        # Evaluate the aggregation flag at call time so env changes take effect even though
        # producers are instantiated at import time.
        disable_aggregation = os.getenv("SYNTHETIC_DISABLE_AGGREGATION", "false").lower() in {"1", "true", "yes", "on"}
        if disable_aggregation:
            return findings
        if all("aggregated_count" in (f.get("metadata") or {}) for f in findings):
            return findings
        clusters = self._cluster_similar_findings(findings)
        return [self._create_aggregate_finding(cluster) for cluster in clusters]

    def _cluster_similar_findings(self, findings: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group findings by similarity (severity, category, metadata patterns)."""
        clusters: Dict[str, List[Dict[str, Any]]] = {}
        for finding in findings:
            key = self._get_cluster_key(finding)
            clusters.setdefault(key, []).append(finding)
        return list(clusters.values())

    def _get_cluster_key(self, finding: Dict[str, Any]) -> str:
        """Generate cluster key based on finding characteristics."""
        category = finding.get("category", self.scanner_name)
        severity = finding.get("severity", "unknown")
        return f"{category}_{severity}_{self._get_pattern(finding)}"

    def _create_aggregate_finding(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a single dense finding from a cluster."""
        base = cluster[0]
        return {
            **base,
            "title": f"Aggregate: {len(cluster)} {base.get('category', self.scanner_name)} findings",
            "description": self._generate_aggregate_description(cluster),
            "metadata": {
                **base.get("metadata", {}),
                "aggregated_count": len(cluster),
                "individual_findings": [f.get("id") for f in cluster],
                "pattern_summary": self._summarize_patterns(cluster),
                "severity_distribution": self._get_severity_dist(cluster),
            },
            "risk_score": max(f.get("risk_score", 0) for f in cluster),
            "probability_actionable": round(
                sum(f.get("probability_actionable", 0.0) for f in cluster) / max(len(cluster), 1),
                3,
            ),
        }

    def _get_pattern(self, finding: Dict[str, Any]) -> str:
        """Extract a lightweight semantic pattern key."""
        metadata = finding.get("metadata", {}) or {}
        pattern_parts = [
            metadata.get("type"),
            metadata.get("pattern"),
            metadata.get("indicator"),
            metadata.get("command"),
            metadata.get("file_path"),
        ]
        filtered = [str(p) for p in pattern_parts if p]
        return "|".join(filtered) if filtered else "generic"

    def _generate_aggregate_description(self, cluster: List[Dict[str, Any]]) -> str:
        """Generate a compact description summarizing the cluster."""
        categories = {f.get("category", self.scanner_name) for f in cluster}
        severities = Counter(f.get("severity", "unknown") for f in cluster)
        top_sev, top_count = severities.most_common(1)[0]
        return (
            f"Aggregated {len(cluster)} findings across {len(categories)} category(ies). "
            f"Dominant severity: {top_sev} ({top_count} findings)."
        )

    def _summarize_patterns(self, cluster: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize recurring patterns within a cluster."""
        patterns = Counter(self._get_pattern(f) for f in cluster)
        return dict(patterns)

    def _get_severity_dist(self, cluster: List[Dict[str, Any]]) -> Dict[str, int]:
        """Return severity distribution for the cluster."""
        return dict(Counter(f.get("severity", "unknown") for f in cluster))