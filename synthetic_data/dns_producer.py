"""DNS telemetry producer for synthetic security findings."""

from typing import Any, Dict, List
import random
from datetime import datetime, timedelta, timezone
from .base_producer import BaseProducer

class DnsProducer(BaseProducer):
    """Producer that emits DNS-related findings including suspicious domains."""

    def __init__(self) -> None:
        super().__init__("dns")
        self.suspicious_domains = [
            "cnc.example.net",
            "steal-data.co",
            "updates.badcorp.ru",
            "dropper.mal",
            "cryptominer.pool",
        ]
        self.malicious_tlds = [".ru", ".cn", ".tk", ".top", ".xyz"]
        self.distro_profiles = [
            {"name": "Debian", "versions": ["11", "12"], "pkg": "apt", "weight": 0.24},
            {"name": "Ubuntu", "versions": ["20.04", "22.04", "24.04"], "pkg": "apt", "weight": 0.28},
            {"name": "Fedora", "versions": ["38", "39", "40"], "pkg": "dnf", "weight": 0.2},
            {"name": "Arch", "versions": ["rolling"], "pkg": "pacman", "weight": 0.14},
            {"name": "Alpine", "versions": ["3.18", "3.19", "3.20"], "pkg": "apk", "weight": 0.14},
        ]

    def generate_findings(self, count: int = 10) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        base_time = datetime.now(timezone.utc)

        for index in range(count):
            scenario = self._choose_scenario()
            finding = self._generate_dns_finding(scenario, index, base_time)
            findings.append(finding)

        return findings

    def _generate_dns_finding(self, scenario: str, index: int, base_time: datetime) -> Dict[str, Any]:
        if scenario == "normal":
            domain = f"www.example{index}.com"
            severity = "info"
            risk_score = 12
            description = f"Standard DNS query observed for domain {domain}"
        elif scenario == "suspicious":
            domain = random.choice(self.suspicious_domains)
            severity = "medium"
            risk_score = random.randint(45, 65)
            description = f"Frequent DNS lookups to watch-listed domain {domain}"
        elif scenario == "malicious":
            domain = random.choice(self.suspicious_domains)
            severity = "high"
            risk_score = random.randint(75, 95)
            description = f"DNS query to known C2 infrastructure {domain}"
        else:
            domain = f"edge{index}{random.choice(self.malicious_tlds)}"
            severity = "low"
            risk_score = 28
            description = "Edge-case DNS pattern detected"

        metadata: Dict[str, Any] = {
            "domain": domain,
            "ttl": random.randint(30, 600),
            "resolver": random.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9", "192.168.1.1"]),
            "query_type": random.choice(["A", "AAAA", "TXT", "CNAME"]),
            "client_ip": f"10.0.{random.randint(0, 10)}.{random.randint(5, 200)}",
            "query_timestamp": (base_time - timedelta(minutes=random.randint(0, 240))).isoformat(),
            "detections": random.sample(["domain_generation", "dns_tunneling", "fast_flux"], k=random.randint(0, 2)),
            **self._sample_distro_profile(),
        }

        return self._generate_base_finding(
            finding_id=f"dns_{index}_{severity}",
            title=f"DNS activity: {domain}",
            severity=severity,
            risk_score=risk_score,
            base_severity_score=risk_score,
            description=description,
            metadata=metadata,
        )

    def _sample_distro_profile(self) -> Dict[str, Any]:
        weights = [p["weight"] for p in self.distro_profiles]
        profile = random.choices(self.distro_profiles, weights=weights, k=1)[0]
        version = random.choice(profile["versions"])
        kernel_minor = random.randint(1, 12)
        kernel_patch = random.randint(1, 30)
        return {
            "distro": profile["name"],
            "distro_version": version,
            "package_manager": profile["pkg"],
            "kernel": f"6.{kernel_minor}.{kernel_patch}-{profile['name'].lower()}",
        }
