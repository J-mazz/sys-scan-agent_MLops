"""Lightweight justification and knowledge-base helpers for synthetic findings.

These helpers avoid external dependencies (no Jinja install) while providing
structured rationales and KB references based on available metadata.
"""
from __future__ import annotations

from typing import Dict, Any, List


def _fmt(val: Any) -> str:
    return str(val) if val is not None else "unknown"


def ensure_kb_refs(finding: Dict[str, Any]) -> None:
    """Attach KB references and hints if missing."""
    metadata = finding.setdefault("metadata", {}) or {}
    if metadata.get("kb_refs"):
        return

    category = finding.get("category", "generic")
    severity = finding.get("severity", "info")

    kb_refs: List[str] = []
    if category in {"network", "dns"}:
        kb_refs.append("MITRE ATT&CK: TA0011 Command and Control")
        kb_refs.append("CIS 9.2: Ensure network egress filtering is implemented")
    elif category in {"processes", "ioc", "endpoint_behavior"}:
        kb_refs.append("MITRE ATT&CK: TA0002 Execution / T1059")
        kb_refs.append("CIS 5: Secure configuration for software on network devices")
    elif category in {"kernel_params", "world_writable", "suid"}:
        kb_refs.append("CIS 1: Inventory and control of enterprise assets")
        kb_refs.append("Linux Hardening: sysctl and file permission baselines")
    else:
        kb_refs.append("General hardening: principle of least privilege")

    if severity in {"high", "critical"}:
        kb_refs.append("Incident triage: prioritize containment within 1h")

    metadata["kb_refs"] = kb_refs


def build_rationale(finding: Dict[str, Any]) -> str:
    """Construct a concise, structured rationale from finding metadata."""
    category = finding.get("category", "generic")
    severity = finding.get("severity", "info")
    metadata = finding.get("metadata", {}) or {}

    pieces: List[str] = []
    pieces.append(f"{severity.title()} {category} finding generated from synthetic knowledge base.")

    indicator = metadata.get("indicator")
    indicator_type = metadata.get("indicator_type")
    indicator_ctx = metadata.get("context")
    command = metadata.get("command") or metadata.get("process")
    distro = metadata.get("distro")
    kernel = metadata.get("kernel_version") or metadata.get("kernel")

    if indicator:
        pieces.append(
            f"Observed {indicator_type or 'indicator'} '{_fmt(indicator)}' ({indicator_ctx or 'no context'}) during activity.")
    if command:
        pieces.append(f"Process/command: {_fmt(command)}.")
    if distro:
        pieces.append(f"Host distro: {_fmt(distro)} {metadata.get('distro_version', '')}.")
    if kernel:
        pieces.append(f"Kernel hint: {_fmt(kernel)}.")

    state_flags = []
    for flag in ["pattern_match", "deleted_executable", "world_writable_executable"]:
        if str(metadata.get(flag, "")).lower() in {"true", "1", "yes"}:
            state_flags.append(flag)
    if state_flags:
        pieces.append(f"Flags: {', '.join(state_flags)} present.")

    kb_refs = metadata.get("kb_refs") or []
    if kb_refs:
        pieces.append(f"KB refs: {', '.join(kb_refs[:3])}.")

    if len(pieces) == 1:
        pieces.append("No additional metadata provided; synthetic rationale fallback.")

    return " ".join(pieces)
