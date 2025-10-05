"""Endpoint behavior analytics producer for host-based anomalies."""

from typing import Any, Dict, List
import random
from datetime import datetime, timedelta
from base_producer import BaseProducer

class EndpointBehaviorProducer(BaseProducer):
    """Generates endpoint behavior analytics findings that capture anomalies."""

    def __init__(self) -> None:
        super().__init__("endpoint_behavior")
        self.user_roles = ["developer", "finance", "it_admin", "sales", "contractor"]
        self.activity_types = ["process_spawn", "login", "file_modification", "privilege_escalation"]

    def generate_findings(self, count: int = 10) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        base_time = datetime.utcnow()

        for index in range(count):
            scenario = self._choose_scenario()
            finding = self._generate_behavior_finding(scenario, index, base_time)
            findings.append(finding)

        return findings

    def _generate_behavior_finding(self, scenario: str, index: int, base_time: datetime) -> Dict[str, Any]:
        user = f"user{random.randint(1000, 1099)}"
        role = random.choice(self.user_roles)
        activity = random.choice(self.activity_types)
        minutes_offset = random.randint(0, 360)
        timestamp = (base_time - timedelta(minutes=minutes_offset)).isoformat()

        if scenario == "normal":
            severity = "info"
            risk = 15
            description = f"Baseline activity observed for {user} ({role})"
        elif scenario == "suspicious":
            severity = "medium"
            risk = random.randint(50, 65)
            description = f"Unusual {activity} pattern detected for {user}"
        elif scenario == "malicious":
            severity = "high"
            risk = random.randint(80, 98)
            description = f"Likely compromised account {user} performing malicious {activity}"
        else:
            severity = "low"
            risk = 30
            description = f"Endpoint behavior edge case identified for {user}"

        metadata: Dict[str, Any] = {
            "user": user,
            "role": role,
            "activity": activity,
            "host": f"host-{random.randint(1, 30):02d}",
            "sequence_length": random.randint(1, 8),
            "timestamp": timestamp,
            "time_delta_minutes": minutes_offset,
            "context": random.choice([
                "after_hours_activity",
                "new_location",
                "rare_binary",
                "sensitive_resource",
                "baseline_pattern",
            ]),
        }

        return self._generate_base_finding(
            finding_id=f"endpoint_behavior_{index}_{user}",
            title=f"Endpoint behavior anomaly for {user}",
            severity=severity,
            risk_score=risk,
            base_severity_score=risk,
            description=description,
            metadata=metadata,
        )
