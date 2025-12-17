import json
import textwrap
from typing import Any, Dict, List
from dataclasses import dataclass
from pathlib import Path

# Mock Config
@dataclass
class MockConfig:
    max_findings_in_prompt: int = 3
    max_correlations_in_prompt: int = 2

config = MockConfig()

# Helpers from notebook
GROUND_TRUTH_DEFAULTS = {
    "version": "ground_truth_v1",
    "enriched_findings": [],
    "correlations": [],
    "reductions": {},
    "summaries": {},
    "actions": [],
}

def _clone_default(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value

def canonicalize_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = record.get("ground_truth") or record.get("data") or record
    canonical = dict(payload)
    for key, default in GROUND_TRUTH_DEFAULTS.items():
        current = canonical.get(key)
        if current is None:
            canonical[key] = _clone_default(default)
        elif key == "version" and not isinstance(current, str):
            canonical[key] = str(current)
            
    # Ensure enriched_findings is a list
    if not isinstance(canonical.get("enriched_findings"), list):
        canonical["enriched_findings"] = []
        
    return canonical

def summarize_findings(payload: Dict[str, Any], limit: int) -> str:
    rows = []
    for finding in (payload.get("enriched_findings") or [])[:limit]:
        title = finding.get("title", "(untitled)")
        severity = finding.get("severity", "unknown")
        risk = finding.get("risk_score", "?")
        rows.append(f"- [{severity}] {title} (risk_score={risk})")
    if not rows:
        rows.append("- No enriched findings present in this slice.")
    return "\n".join(rows)

def summarize_correlations(payload: Dict[str, Any], limit: int) -> str:
    rows = []
    for corr in (payload.get("correlations") or [])[:limit]:
        title = corr.get("title", "(untitled)")
        related = ", ".join((corr.get("related_finding_ids") or [])[:3]) or "n/a"
        rows.append(f"- {title} → related: {related}")
    if not rows:
        rows.append("- No correlations linked in this slice.")
    return "\n".join(rows)

def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "query_baseline",
            "description": "Check if a finding is new or existing in the baseline database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "finding_id": {"type": "string"},
                    "title": {"type": "string"},
                    "severity": {"type": "string"}
                },
                "required": ["finding_id"]
            }
        }
    ]

def build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = canonicalize_ground_truth(record)
    summaries = payload.get("summaries") or {}
    exec_summary = summaries.get("executive_summary") or "No executive summary provided."
    triage_summary = summaries.get("triage_summary") or "No triage summary provided."
    
    tools_json = json.dumps(get_tool_definitions(), indent=2)
    
    system_prompt = textwrap.dedent(f"""
        You are a senior security analytics engineer.
        
        Your capabilities:
        1. Reason step-by-step about synthetic host and network telemetry.
        2. Use available tools to verify findings against baselines.
        3. Emit a final JSON that conforms to the sys-scan ground_truth schema (version ground_truth_v1).
        
        Available Tools:
        {tools_json}
        
        Output Format:
        <think>
        [Your reasoning process here]
        </think>
        <tools>
        [Optional: Tool calls if needed, e.g. {{"name": "query_baseline", "arguments": {{...}}}}]
        </tools>
        <answer>
        [Final JSON object]
        </answer>
        
        Always prefer defensive mitigations, never offensive guidance.
    """).strip()

    user_prompt = textwrap.dedent(f"""
        Review the following synthetic security telemetry and produce a final ground truth JSON.

        Top findings (capped at {config.max_findings_in_prompt}):
        {summarize_findings(payload, config.max_findings_in_prompt)}

        Correlations (capped at {config.max_correlations_in_prompt}):
        {summarize_correlations(payload, config.max_correlations_in_prompt)}

        Executive summary: {exec_summary}
        Triage summary: {triage_summary}

        Respond with the required XML tags: <think>, <tools> (if applicable), and <answer>.
    """).strip()
    
    thought_content = f"Analyzing {len(payload.get('enriched_findings', []))} findings. Executive summary indicates: {exec_summary[:100]}..."
    
    assistant_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    
    assistant_content = f"<think>\n{thought_content}\n</think>\n<tools>\n[]\n</tools>\n<answer>\n{assistant_json}\n</answer>"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]

# Test Execution
def run_test():
    example_path = Path("synthetic_data/synthetic_dataset_example.json")
    if not example_path.exists():
        print(f"Error: {example_path} not found.")
        return

    with open(example_path, "r") as f:
        data = json.load(f)
    
    print("Loaded example data.")
    messages = build_messages(data)
    
    print("\n--- System Prompt ---")
    print(messages[0]["content"][:500] + "...")
    
    print("\n--- User Prompt ---")
    print(messages[1]["content"])
    
    print("\n--- Assistant Response ---")
    content = messages[2]["content"]
    print(content[:500] + "...")
    
    # Verify tags
    if "<think>" in content and "<tools>" in content and "<answer>" in content:
        print("\n✅ All required tags present.")
    else:
        print("\n❌ Missing tags.")

    # Verify JSON parsing
    try:
        start = content.find("<answer>") + len("<answer>")
        end = content.find("</answer>")
        json_str = content[start:end].strip()
        json.loads(json_str)
        print("✅ JSON in <answer> tag is valid.")
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")

if __name__ == "__main__":
    run_test()
