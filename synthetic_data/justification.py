"""
Enhanced justification engine that generates 'Reasoning Traces' 
optimized for <think> block training.
"""
from __future__ import annotations
from typing import Dict, Any, List
import random


def _fmt(val: Any) -> str:
    return str(val) if val is not None else "unknown"


def build_rationale(finding: Dict[str, Any]) -> str:
    """
    Construct a structured 'Chain of Thought' rationale using causal connectors.
    
    This introduces the vocabulary needed for GRPO (because, therefore, however)
    directly into the training data.
    """
    category = finding.get("category", "generic")
    severity = finding.get("severity", "info").lower()
    metadata = finding.get("metadata", {}) or {}
    title = finding.get("title", "Unknown Finding")

    # 1. THE PREMISE (Observation)
    trace = [f"I am analyzing a {severity} severity finding related to {category}."]
    
    # 2. THE CAUSAL LINK (The "Because")
    if category == "network":
        port = metadata.get("port")
        proto = metadata.get("protocol")
        trace.append(f"The risk score is elevated **because** port {_fmt(port)}/{_fmt(proto)} is exposed, which increases the attack surface.")
    elif category == "suid":
        # Prefer explicit naming keys, fall back to title when missing
        file = metadata.get("file_path") or metadata.get("binary") or metadata.get("path") or title
        trace.append(f"This is flagged **because** the binary '{_fmt(file)}' has the SUID bit set.")
    elif category == "process":
        proc = metadata.get("process_name") or metadata.get("command")
        user = metadata.get("user")
        trace.append(f"The anomaly detection triggered **because** process '{_fmt(proc)}' was spawned by user '{_fmt(user)}' in an unusual context.")
    elif category == "kernel_params":
        # Use the explicit param if present, otherwise fall back to the finding title
        param = metadata.get("param") or title
        trace.append(f"The configuration is non-compliant **because** the kernel parameter '{_fmt(param)}' deviates from the hardened baseline.")
    else:
        trace.append(f"The alert was triggered **because** the system state matches known patterns for {title}.")

    # 3. THE IMPLICATION (The "Implies/Impact")
    if severity in ["high", "critical"]:
        trace.append("This **implies** an immediate threat to system integrity that could allow an attacker to gain unauthorized access.")
    elif severity == "medium":
        trace.append("This **implies** a misconfiguration that reduces the system's defense-in-depth posture.")
    else:
        trace.append("The **impact** is currently limited to informational visibility.")

    # 4. THE COUNTER-FACTUAL / NUANCE (The "However")
    if metadata.get("container_id"):
        trace.append("**However**, this process is running inside a container, which limits the blast radius of a potential exploit.")
    elif metadata.get("is_trusted_user") or metadata.get("user") == "root":
        trace.append("**However**, if this activity is part of a scheduled maintenance window, it might be a false positive.")
    else:
        trace.append("**However**, verification against the specific host baseline is recommended to confirm context.")

    # 5. THE CONCLUSION (The "Therefore/Remediation")
    trace.append(f"**Therefore**, I have classified this as {severity}.")
    
    return " ".join(trace)


def ensure_kb_refs(finding: Dict[str, Any]) -> None:
    """Attach KB references and hints if missing."""
    metadata = finding.setdefault("metadata", {}) or {}
    if metadata.get("kb_refs"):
        return

    category = finding.get("category", "generic")
    kb_refs: List[str] = []
    if category in {"network", "dns"}:
        kb_refs.append("MITRE ATT&CK: TA0011 Command and Control")
    elif category in {"processes", "ioc"}:
        kb_refs.append("MITRE ATT&CK: TA0002 Execution")
    else:
        kb_refs.append("General hardening: principle of least privilege")

    metadata["kb_refs"] = kb_refs