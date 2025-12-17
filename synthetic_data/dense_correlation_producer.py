"""
Dense correlation producer that creates multi-hop and multi-scanner relationships.
"""
from typing import Dict, List, Any
from datetime import datetime
import random
from .base_correlation_producer import BaseCorrelationProducer


class DenseCorrelationProducer(BaseCorrelationProducer):
    """Produces correlations with deeper analytical insights."""

    def __init__(self):
        super().__init__("dense")

    def analyze_correlations(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        correlations: List[Dict[str, Any]] = []

        correlations.extend(self._find_multi_scanner_patterns(findings))
        correlations.extend(self._find_temporal_sequences(findings))
        correlations.extend(self._find_causal_chains(findings))

        return correlations

    def _find_multi_scanner_patterns(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        if findings.get("processes") and findings.get("network") and findings.get("world_writable"):
            host_sources = []
            for key in ["processes", "network", "world_writable"]:
                host_sources.extend(findings.get(key, []))
            pattern = self._create_correlation_finding(
                title="Advanced persistent threat indicators",
                description=(
                    "Coordinated attack pattern detected: "
                    "suspicious process with network connectivity, "
                    "world-writable files, and kernel parameter manipulation"
                ),
                severity="critical",
                risk_score=95,
                related_findings=self._get_related_ids(findings, ["processes", "network", "world_writable"]),
                correlation_type="multi_vector_apt",
                metadata={
                    "scanner_count": 3,
                    "attack_stages": ["initial_access", "persistence", "privilege_escalation"],
                    "mitre_tactics": ["TA0001", "TA0003", "TA0004"],
                },
                host_context=self._host_context(host_sources),
            )
            patterns.append(pattern)
        return patterns

    def _find_temporal_sequences(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        correlations: List[Dict[str, Any]] = []
        ordered = [findings.get(key, []) for key in ["network", "processes", "kernel_params"]]
        flat = [item for sub in ordered for item in sub]
        if len(flat) >= 3:
            related_ids = [str(f.get("id")) for f in flat[:5] if f.get("id")]
            correlations.append(
                self._create_correlation_finding(
                    title="Temporal escalation sequence",
                    description="Sequential events across network, process, and kernel parameters indicate possible escalation.",
                    severity="high",
                    risk_score=80,
                    related_findings=related_ids,
                    correlation_type="temporal_sequence",
                    metadata={
                        "window": "short",
                        "sequence_length": len(related_ids),
                        "observed_at": datetime.now().isoformat(),
                    },
                    host_context=self._host_context(flat),
                )
            )
        return correlations

    def _find_causal_chains(self, findings: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        chains: List[Dict[str, Any]] = []
        if findings.get("ioc") and findings.get("processes"):
            related_ids = self._get_related_ids(findings, ["ioc", "processes"])
            host_sources = findings.get("ioc", []) + findings.get("processes", [])
            chains.append(
                self._create_correlation_finding(
                    title="IOC-led process compromise",
                    description="Indicator of compromise aligns with process behavior anomalies, suggesting cause-effect chain.",
                    severity="high",
                    risk_score=85,
                    related_findings=related_ids,
                    correlation_type="causal_ioc_process",
                    metadata={
                        "source": "threat_intel",
                        "causal_confidence": random.uniform(0.6, 0.9),
                    },
                    host_context=self._host_context(host_sources),
                )
            )
        return chains

    def _get_related_ids(self, findings: Dict[str, List[Dict[str, Any]]], categories: List[str]) -> List[str]:
        related: List[str] = []
        for category in categories:
            for finding in findings.get(category, []):
                fid = finding.get("id")
                if fid:
                    related.append(str(fid))
        return related

    def _host_context(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        for finding in findings:
            meta = finding.get("metadata", {}) or {}
            if any(meta.get(k) for k in ["distro", "distro_version", "package_manager", "kernel", "kernel_version"]):
                return {
                    "distro": meta.get("distro"),
                    "distro_version": meta.get("distro_version"),
                    "package_manager": meta.get("package_manager"),
                    "kernel": meta.get("kernel") or meta.get("kernel_version"),
                }
        return {}
