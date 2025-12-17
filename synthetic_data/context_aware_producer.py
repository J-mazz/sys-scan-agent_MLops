"""
Context-aware producer that generates findings with richer relationships.
"""
from typing import Dict, List, Any
import random
import uuid
from .base_producer import BaseProducer


class ContextAwareProducer(BaseProducer):
    """Generates findings with built-in contextual relationships."""

    def __init__(self, scanner_name: str = "context"):
        super().__init__(scanner_name)
        self.context_state: Dict[str, Any] = {}

    def generate_findings(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generate contextually-aware findings."""
        attack_chain = self._generate_attack_chain()
        environment = self._generate_environment_context()

        findings: List[Dict[str, Any]] = []
        for scenario in self._select_scenarios(count):
            findings.append(self._generate_contextual_finding(scenario, attack_chain, environment))
        return findings

    def _select_scenarios(self, count: int) -> List[str]:
        return [self._choose_scenario() for _ in range(count)]

    def _generate_attack_chain(self) -> Dict[str, Any]:
        """Generate a coherent attack narrative."""
        chains = [
            {
                "name": "privilege_escalation",
                "stages": ["reconnaissance", "initial_access", "privilege_escalation", "persistence"],
                "indicators": ["suspicious_suid", "kernel_exploit", "cron_backdoor"],
            },
            {
                "name": "data_exfiltration",
                "stages": ["credential_theft", "lateral_movement", "data_staging", "exfiltration"],
                "indicators": ["suspicious_network", "unusual_process", "large_transfer"],
            },
        ]
        chosen = random.choice(chains)
        self.context_state["attack_chain"] = chosen
        return chosen

    def _generate_environment_context(self) -> Dict[str, Any]:
        """Simulate environmental backdrop (hosts, users, roles)."""
        environment = {
            "host": random.choice(["web-01", "db-02", "k8s-node-3", "jump-box"]),
            "user": random.choice(["root", "svc_backup", "analyst", "developer"]),
            "network_segment": random.choice(["prod", "dmz", "staging", "corp"]),
            "time_window": random.choice(["business_hours", "after_hours", "weekend"]),
        }
        self.context_state["environment"] = environment
        return environment

    def _generate_contextual_finding(
        self,
        scenario: str,
        attack_chain: Dict[str, Any],
        environment: Dict[str, Any],
    ) -> Dict[str, Any]:
        stage = random.choice(attack_chain.get("stages", ["observation"]))
        indicator = random.choice(attack_chain.get("indicators", ["anomaly"]))
        severity_map = {
            "normal": "info",
            "suspicious": "medium",
            "malicious": "high",
            "edge_case": random.choice(["low", "medium"]),
        }
        severity = severity_map.get(scenario, "medium")
        risk_score = {
            "info": 15,
            "low": 25,
            "medium": 55,
            "high": 80,
            "critical": 95,
        }.get(severity, 40)

        finding_id = f"ctx_{uuid.uuid4().hex[:10]}"
        title = f"{attack_chain['name']} stage: {stage}"
        description = (
            f"{indicator} detected on {environment['host']} during {stage}. "
            f"User {environment['user']} on {environment['network_segment']} segment."
        )

        metadata = {
            "attack_chain": attack_chain,
            "stage": stage,
            "indicator": indicator,
            "environment": environment,
            "scenario": scenario,
        }

        return self._generate_base_finding(
            finding_id=finding_id,
            title=title,
            severity=severity,
            risk_score=risk_score,
            base_severity_score=risk_score,
            description=description,
            metadata=metadata,
        )
