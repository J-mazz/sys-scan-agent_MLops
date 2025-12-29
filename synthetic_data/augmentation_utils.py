"""Utilities to increase diversity in synthetic findings.

Provides randomizers and perturbation helpers used by producers to broaden
value distributions and make deduplication less aggressive.
"""
from __future__ import annotations

import random
import uuid
from typing import List


def unique_token(prefix: str = "t") -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"


def random_port(include_common: bool = True) -> int:
    """Return a port with weighted probability across common, suspicious, and ephemeral ranges."""
    r = random.random()
    if r < 0.45 and include_common:
        # common well-known ports
        return random.choice([22, 80, 443, 25, 110, 143, 993, 995, 3306, 5432, 6379, 27017])
    elif r < 0.75:
        # suspicious/malware-associated ports
        return random.choice([4444, 1337, 6667, 31337, 12345, 54321, 3389, 8088, 9001, 9050, 2222, 50050])
    else:
        # ephemeral range
        return random.randint(1024, 65535)


def random_ja3() -> str:
    # simple synthetic JA3 generator with some variance
    a = random.choice([769, 771, 771, 771, 771, 771, 770])
    b = ",".join(str(x) for x in random.sample([4865,4866,4867,49195,49196,52393,52392,51000,49197], k=4))
    c = ",".join(str(x) for x in random.sample([0,10,11,23,65281,29,23,24], k=4))
    d = ",".join(str(x) for x in random.sample([29,23,24,0], k=2))
    e = random.choice(["0", ""])
    return f"{a},{b},{c},{d},{e}"


def random_sni() -> str:
    subs = ["api", "svc", "update", "metrics", "auth", "vault", "cdn"]
    domains = ["internal", "service", "example.com", "malicious.test", "update.windows.com"]
    return f"{random.choice(subs)}.{random.choice(domains)}"


def random_exfil_destination() -> str:
    templates = [
        "s3-malicious-{n}.s3.amazonaws.com",
        "gdrive-{n}.badshare.com",
        "pastebin.com/raw/{n}",
        "transfer.sh/{n}",
        "ipfs.io/ipfs/{n}",
    ]
    t = random.choice(templates)
    return t.format(n=uuid.uuid4().hex[:12])


def perturb_text(text: str) -> str:
    """Add small perturbations to a text string to increase uniqueness."""
    if random.random() < 0.25:
        return f"{text} ({unique_token('x')})"
    if random.random() < 0.15:
        return f"{text} - {random.choice(['detected', 'observed', 'flagged'])}"
    return text


def random_username() -> str:
    prefixes = ["user", "svc", "admin", "backup", "deploy", "ops"]
    return f"{random.choice(prefixes)}{random.randint(1,9999)}"
