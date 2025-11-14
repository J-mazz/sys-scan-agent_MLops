"""LangChain correlation bridge executed under a Python 3.12 runtime.

This module is imported by the main pipeline (which may run on a Python 3.14
free-threaded build) without pulling in LangChain dependencies. When executed as
``python -m synthetic_data.langchain_bridge`` it expects a JSON payload on
stdin and emits the correlation payload JSON on stdout.

Inputs (JSON object):
    {
        "candidates": [ {"id": "...", "title": "...", ...}, ... ],
        "model": "gpt-4o-mini",
        "temperature": 0.1
    }

Outputs (JSON object):
    Correlation payload with the schema enforced by
    :class:`LangChainCorrelationProducer`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

try:
    from jinja2 import Environment, Template  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Environment = None  # type: ignore
    Template = None  # type: ignore

PROMPT_TEMPLATE = (
    "You are a SOC analyst creating a high-value correlation. "
    "Use the provided findings to write ONE correlation in strict JSON with fields: "
    "title (string), description (string), severity (info|low|medium|high|critical), "
    "risk_score (integer 0-100), correlation_type (string), insight (string summarising the link). "
    "Ensure you reference at least two finding IDs in a field named related_ids (array of strings)."
    "\n\nContext:\n{context}\n\nJSON:"
)

_JINJA_SOURCE = """
You are a SOC analyst creating a high-value correlation.
Use the provided findings to write ONE correlation in strict JSON with fields:
- title (string)
- description (string)
- severity (info|low|medium|high|critical)
- risk_score (integer 0-100)
- correlation_type (string)
- insight (string summarising the link)

Ensure you reference at least two finding IDs in a field named related_ids (array of strings).
Guardrails:
- Never hallucinate IDs; only use those provided below.
- Choose the highest justified severity among the referenced findings.
- Risk scores must be integers between 0 and 100.
- Respond with JSON only.

Context:
{% for item in candidates %}- {{ item.id }} ({{ item.severity|default("info") }}, {{ item.risk_score|default(0) }}): {{ item.title|default("") }}
{% endfor %}

JSON:
"""

if Environment is not None:  # pragma: no branch - simple import guard
    _JINJA_ENV = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
    _JINJA_TEMPLATE: Template | None = _JINJA_ENV.from_string(_JINJA_SOURCE)
else:  # pragma: no cover - optional dependency missing
    _JINJA_TEMPLATE = None


def build_context_lines(candidates: List[Dict[str, Any]]) -> str:
    """Create the context string passed into the LangChain prompt."""

    lines = []
    for candidate in candidates:
        finding_id = candidate.get("id", "unknown")
        severity = candidate.get("severity", "info")
        risk = candidate.get("risk_score", 0)
        title = candidate.get("title", "")
        lines.append(f"- {finding_id} ({severity}, {risk}): {title}")
    return "\n".join(lines)


def render_prompt(candidates: List[Dict[str, Any]]) -> str:
    """Render the LangChain prompt using Jinja when available."""

    if _JINJA_TEMPLATE is not None:
        rendered = _JINJA_TEMPLATE.render(candidates=candidates)
        return rendered.strip()

    # Fallback to legacy string formatting if Jinja is unavailable.
    context = build_context_lines(candidates)
    return PROMPT_TEMPLATE.format(context=context)


def run_langchain_correlation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the LangChain correlation prompt using the supplied payload."""

    # Import LangChain lazily so the module stays importable under Python 3.14.
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI

    model = payload.get("model", "gpt-4o-mini")
    temperature = float(payload.get("temperature", 0.1))
    candidates = payload["candidates"]

    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=600)

    prompt_text = render_prompt(candidates)
    completion = llm.invoke(prompt_text)
    raw_text = completion.content if hasattr(completion, "content") else str(completion)

    correlation = json.loads(raw_text)

    # Minimal validation mirroring the producer expectations.
    required_fields = {"title", "description", "severity", "risk_score", "related_ids"}
    if not required_fields.issubset(correlation):
        raise ValueError("LangChain response missing required fields")
    if len(correlation.get("related_ids", [])) < 2:
        raise ValueError("LangChain response requires at least two related_ids")

    return correlation


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"Invalid JSON payload: {exc}\n")
        return 1

    try:
        result = run_langchain_correlation(payload)
    except Exception as exc:  # pragma: no cover - handled upstream
        sys.stderr.write(f"LangChain correlation failed: {exc}\n")
        return 2

    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":  # pragma: no cover - integration entrypoint
    sys.exit(main())
