"""Deduplication agent to filter redundant findings while keeping most informative ones."""
from typing import List, Dict, Any
import difflib
import os
import hashlib


class DeduplicationAgent:
    """Removes redundant findings, keeping only the most informative."""

    def __init__(self) -> None:
        # Allow tuning via environment: higher threshold = less aggressive dedup
        try:
            self.similarity_threshold = float(os.getenv("SYNTHETIC_DEDUP_THRESHOLD", "0.95"))
        except ValueError:
            self.similarity_threshold = 0.95

    def deduplicate(
        self,
        findings: List[Dict[str, Any]],
        similarity_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Remove redundant findings based on semantic pattern similarity."""
        if not findings:
            return []

        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        unique_findings: List[Dict[str, Any]] = []
        seen_patterns: List[str] = []

        sorted_findings = sorted(
            findings,
            key=lambda f: f.get("risk_score", 0),
            reverse=True,
        )

        for finding in sorted_findings:
            pattern = self._extract_pattern(finding)
            if not self._is_similar_to_seen(pattern, seen_patterns, threshold):
                unique_findings.append(finding)
                seen_patterns.append(pattern)

        return unique_findings

    def _extract_pattern(self, finding: Dict[str, Any]) -> str:
        """Extract semantic pattern from finding.

        To avoid over-collapsing sparse findings, include title/description hashes when metadata is thin.
        """
        metadata = finding.get("metadata", {}) or {}
        category = finding.get("category", "unknown")
        severity = finding.get("severity", "unknown")

        meta_tokens = [
            str(metadata.get("type", "")),
            str(metadata.get("pattern", "")),
            str(metadata.get("indicator", "")),
            str(metadata.get("command", "")),
            str(metadata.get("file_path", "")),
        ]

        base_pattern = f"{category}:{severity}:{'|'.join(meta_tokens)}"

        # If metadata is too sparse, append stable hashes of title/description to reduce false merges
        if all(not t for t in meta_tokens):
            title = finding.get("title", "")
            desc = finding.get("description", "")
            digest = hashlib.md5(f"{title}|{desc}".encode("utf-8")).hexdigest()[:8]
            base_pattern = f"{base_pattern}:{digest}"

        return base_pattern

    def _is_similar_to_seen(self, pattern: str, seen_patterns: List[str], threshold: float) -> bool:
        for seen in seen_patterns:
            if difflib.SequenceMatcher(None, pattern, seen).ratio() >= threshold:
                return True
        return False
