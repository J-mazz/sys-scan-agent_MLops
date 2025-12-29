"""Network scanner producer for generating synthetic network-related findings."""

from typing import Any, Dict, List
import random
import uuid

from .base_producer import BaseProducer


class NetworkProducer(BaseProducer):
    """Producer for synthetic network scanner findings."""

    def __init__(self):
        super().__init__("network")
        self.common_ports = [
            22,
            80,
            443,
            3306,
            5432,
            6379,
            27017,
            25,
            110,
            143,
            993,
            995,
            8080,
            8443,
            9200,
            11211,
        ]
        self.suspicious_ports = [4444, 1337, 6667, 31337, 12345, 54321, 3389, 8088, 9001, 9050, 2222, 50050]
        self.protocols = ["tcp", "udp", "tcp6", "udp6"]
        self.common_services = [
            (80, "http", "nginx"),
            (443, "https", "envoy"),
            (22, "ssh", "openssh"),
            (3306, "mysql", "mysqld"),
            (5432, "postgres", "postgres"),
            (6379, "redis", "redis-server"),
            (27017, "mongodb", "mongod"),
            (9200, "elasticsearch", "java"),
            (11211, "memcached", "memcached"),
            (25, "smtp", "postfix"),
            (8080, "http-alt", "jetty"),
            (8443, "https-alt", "tomcat"),
        ]
        self.c2_hosts = [
            "198.51.100.{octet}",
            "203.0.113.{octet}",
            "45.67.230.{octet}",
            "77.247.110.{octet}",
            "185.220.101.{octet}",
            "darknode{n}.onion",
            "c2-drop.{n}.evilcdn.net",
        ]
        self.exfil_hosts = [
            "s3-malicious-{n}.s3.amazonaws.com",
            "gdrive-{n}.badshare.com",
            "pastebin.com/raw/{n}",
            "transfer.sh/{n}",
            "ipfs.io/ipfs/{n}",
        ]
        self.ja3_fingerprints = [
            "769,4865-4866-4867-49195-49196-52393-52392,0-23-65281,29-23-24,0",
            "771,4865-4866-4867-49195,0-10-11,23-65281,29-23-24,0",
        ]
        self.sni_hosts = [
            "api.internal",
            "k8s.cluster.local",
            "vault.service",
            "update.windows.com",
            "telemetry.apple.com",
        ]

    def generate_findings(self, count: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic network findings."""
        findings = []

        for i in range(count):
            scenario = self._choose_scenario()
            finding = self._generate_network_finding(scenario, i)
            findings.append(finding)

        return findings

    def _generate_network_finding(self, scenario: str, index: int) -> Dict[str, Any]:
        """Generate a single network finding based on scenario."""

        if scenario == "normal":
            return self._generate_normal_network(index)
        elif scenario == "suspicious":
            return self._generate_suspicious_network(index)
        elif scenario == "malicious":
            return self._generate_malicious_network(index)
        elif scenario == "edge_case":
            return self._generate_edge_case_network(index)
        else:
            return self._generate_normal_network(index)

    def _generate_normal_network(self, index: int) -> Dict[str, Any]:
        """Generate a normal network finding with slight perturbations to increase uniqueness."""
        from .augmentation_utils import random_port, random_sni, perturb_text

        port = random_port(include_common=True)
        protocol = random.choice(["tcp", "tcp6", "udp"])  # include udp more often
        state = random.choice(["LISTEN", "ESTABLISHED", "CLOSE_WAIT"])

        title = f"Normal network service on port {port}"
        title = perturb_text(title)

        return self._generate_base_finding(
            finding_id=f"net_normal_{port}_{index}_{random.randint(0,9999)}",
            title=title,
            severity="info",
            risk_score=10,
            base_severity_score=10,
            description=f"{perturb_text('Standard service listening')} on port {port}/{protocol}",
            metadata={
                "port": port,
                "protocol": protocol,
                "state": state,
                "local_address": f"0.0.0.0:{port}" if protocol in ["tcp", "udp"] else f"[::]:{port}",
                "foreign_address": "0.0.0.0:0"
                if state == "LISTEN"
                else f"192.168.1.{random.randint(1,254)}:{random.randint(1024,65535)}",
                "inode": random.randint(10000, 99999),
                "sni": random_sni() if random.random() < 0.3 else None
            }
        )

    def _generate_suspicious_network(self, index: int) -> Dict[str, Any]:
        """Generate a suspicious network finding with enhanced variability."""
        from .augmentation_utils import random_port, random_ja3, random_sni, unique_token

        port = random_port(include_common=False)
        protocol = random.choice(self.protocols)

        return self._generate_base_finding(
            finding_id=f"net_susp_{port}_{index}_{unique_token()}",
            title=f"Suspicious port {port} open ({unique_token('p')})",
            severity="medium",
            risk_score=60,
            base_severity_score=60,
            description=f"Unusual port {port} is listening, commonly associated with malware",
            metadata={
                "port": port,
                "protocol": protocol,
                "state": "LISTEN",
                "local_address": f"0.0.0.0:{port}",
                "foreign_address": "0.0.0.0:0",
                "inode": random.randint(100000, 999999),
                "process": f"/usr/bin/nc -l {port}" if random.random() < 0.6 else f"/usr/bin/socat - TCP:{unique_token('c')}.example:{port}",
                "c2_host": random.choice(self.c2_hosts).format(
                    octet=random.randint(10, 250), n=random.randint(1, 9999)
                ) if random.random() < 0.8 else None,
                "ja3": random_ja3(),
                "sni": random_sni() if random.random() < 0.5 else None,
                "asn": random.choice([9009, 16276, 202425, 14061, 60111, 49505]),
                "geo": random.choice(["RU", "CN", "IR", "BR", "US", "UA"]),
            }
        )

    def _generate_malicious_network(self, index: int) -> Dict[str, Any]:
        """Generate a malicious network finding with varied fingerprints and exfil targets."""
        from .augmentation_utils import random_ja3, random_exfil_destination, unique_token

        port = random.randint(1, 65535)
        protocol = "tcp"

        return self._generate_base_finding(
            finding_id=f"net_mal_{port}_{index}_{unique_token('m')}",
            title=f"Malicious C2 communication detected ({unique_token('m')})",
            severity="critical",
            risk_score=95,
            base_severity_score=95,
            description=f"Outbound connection to known malicious IP on port {port}",
            metadata={
                "port": port,
                "protocol": protocol,
                "state": "ESTABLISHED",
                "local_address": f"192.168.1.{random.randint(1,254)}:{random.randint(1024,65535)}",
                "foreign_address": f"203.0.113.{random.randint(1,254)}:{port}",
                "inode": random.randint(1000000, 9999999),
                "process": random.choice(["/tmp/.backdoor", f"/var/run/{unique_token('p')}", "/usr/local/bin/evil"]),
                "malicious_ip": True,
                "c2_indicator": True,
                "c2_host": random.choice(self.c2_hosts).format(octet=random.randint(10, 250), n=random.randint(1, 9999)),
                "exfil_destination": random_exfil_destination(),
                "ja3": random_ja3(),
                "sni": random.choice(self.sni_hosts) if random.random() < 0.5 else None,
                "asn": random.choice([49505, 21335, 9009, 16276, 20473, 13335]),
                "geo": random.choice(["RU", "CN", "IR", "UA", "US", "BR"]),
            }
        )

    def _generate_edge_case_network(self, index: int) -> Dict[str, Any]:
        """Generate an edge case network finding."""
        edge_cases = [
            {"port": 0, "desc": "Port 0 (invalid port)"},
            {"port": 65535, "desc": "Maximum port number"},
            {"port": random.randint(1, 65535), "protocol": "unknown", "desc": "Unknown protocol"},
            {"port": 22, "state": "UNKNOWN", "desc": "Unknown socket state"}
        ]
        edge_case = random.choice(edge_cases)

        return self._generate_base_finding(
            finding_id=f"net_edge_{index}",
            title="Network edge case",
            severity="low",
            risk_score=20,
            base_severity_score=20,
            description=edge_case["desc"],
            metadata={
                "port": edge_case.get("port", 0),
                "protocol": edge_case.get("protocol", "tcp"),
                "state": edge_case.get("state", "LISTEN"),
                "local_address": f"127.0.0.1:{edge_case.get('port', 0)}",
                "foreign_address": "0.0.0.0:0",
                "inode": 0 if edge_case.get("port") == 0 else random.randint(10000, 99999),
                "asn": random.choice([0, 13335, 16509, 32934]),
                "geo": random.choice(["--", "US", "DE", "SG"]),
            }
        )