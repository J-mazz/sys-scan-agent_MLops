# data_orchestrator.py
import json, math, glob, os, random, itertools
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# ---------- Utilities ----------
def softplus(x): return math.log1p(math.exp(x))

def zscore(x, mean, std): return (x - mean) / (std + 1e-8)

def try_num(x, default=0.0):
    if isinstance(x,(int,float)): return float(x)
    if isinstance(x,str):
        try: return float(x)
        except:
            table = {"low":0, "medium":1, "med":1, "high":2, "critical":3, "ok":1, "good":2, "great":3}
            return float(table.get(x.lower(), default))
    return default

# ---------- Read synthetic JSON blobs ----------
def load_synthetic_blobs(path="synthetic_data"):
    files = sorted(glob.glob(os.path.join(path, "**/*.json"), recursive=True) +
                   glob.glob(os.path.join(path, "**/*.jsonl"), recursive=True))
    blobs = []
    for f in files:
        if f.endswith(".jsonl"):
            with open(f) as fh:
                for line in fh:
                    line=line.strip()
                    if line: blobs.append(json.loads(line))
        else:
            with open(f) as fh:
                doc = json.load(fh)
                if isinstance(doc, list): blobs.extend(doc)
                else: blobs.append(doc)
    return blobs

# ---------- Extract correlation strength ρ ----------
# Priority: explicit LangChain correlation if present; else graph-based proxy.
def correlation_strength(item: Dict[str,Any]) -> float:
    # 1) Direct signal from LangChainCorrelationProducer
    corr = item.get("correlations") or []
    # try to find a field like {"type":"langchain", "score": 0.72}
    rho = None
    for c in corr:
        t = (c.get("type") or c.get("producer") or "").lower()
        if "langchain" in t:
            if "score" in c: rho = try_num(c["score"], 0.0)
            elif "corr" in c: rho = try_num(c["corr"], 0.0)
            elif "strength" in c: rho = try_num(c["strength"], 0.0)
            break
    if rho is not None:
        return max(-1.0, min(1.0, float(rho)))

    # 2) Proxy: normalize degree/weight from the correlations array
    # Heuristic: more diverse, stronger edges -> higher rho
    deg = len(corr)
    if deg == 0: return 0.0
    weights = []
    for c in corr:
        for k in ("score","weight","strength","corr"):
            if k in c:
                weights.append(abs(try_num(c[k], 0.0)))
                break
    if not weights:
        # degree-based proxy, soft-clipped
        return max(0.0, min(1.0, 1.0 - math.exp(-0.25*deg)))
    w = sum(weights)/len(weights)
    # squash to [-1,1] (assume non-negative weights)
    return max(0.0, min(1.0, w))

# ---------- Quality label from AdvancedVerificationAgent ----------
def quality_label(item: Dict[str,Any]) -> float:
    qa = item.get("quality") or item.get("verification") or {}
    # look for consolidated score; else combine components
    candidates = [qa.get(k) for k in ("score","quality_score","overall","total","final")]
    for c in candidates:
        if c is not None: return try_num(c, 1.0)
    # fallback: components like realism, coherence, diversity, schema_ok…
    parts = []
    for k in ("realism","coherence","diversity","schema_ok","abundance","consistency"):
        if k in qa: parts.append(try_num(qa[k], 1.0))
    return sum(parts)/len(parts) if parts else 1.0

# ---------- Render INPUT_BLOCK ----------
def render_finding(f: Dict[str,Any]) -> str:
    # Pick useful facets with robust fallbacks
    typ  = f.get("type") or f.get("category") or "unknown"
    sev  = f.get("severity") or f.get("risk") or "unknown"
    host = f.get("host") or f.get("asset") or f.get("node") or "n/a"
    desc = f.get("description") or f.get("details") or f.get("evidence") or ""
    extras = []
    for k in ("process","binary","port","proto","module","path","ioc","command","user"):
        if k in f: extras.append(f"{k}={f[k]}")
    extras_s = (", ".join(extras)) if extras else ""
    return (f"type={typ}; severity={sev}; host={host}\n"
            f"details: {desc}\n{extras_s}").strip()

def render_correlations(corrs: List[Dict[str,Any]], max_items=3) -> str:
    if not corrs: return "no_correlations"
    take = corrs[:max_items]
    lines = []
    for c in take:
        t = c.get("type") or c.get("producer") or "corr"
        s = c.get("score") or c.get("strength") or c.get("weight") or ""
        j = c.get("justification") or c.get("explanation") or ""
        lines.append(f"- {t}: score={s} {j}".strip())
    return "Local correlations:\n" + "\n".join(lines)

# ---------- Build training records ----------
@dataclass
class TrainRec:
    task: str         # TRIAGE_SUMMARY | REMEDIATION_ACTIONS | EXEC_SUMMARY
    input_text: str
    target_text: str
    label_score: float
    corr_rho: float
    group_id: str     # for pair-building if needed

def make_records(blob: Dict[str,Any]) -> List[TrainRec]:
    out = []
    findings = blob.get("enriched_findings") or []
    summaries = blob.get("summaries") or []
    actions   = blob.get("actions") or []
    # Index optional mappings from finding id -> summary/actions
    idx_by_id = {}
    for i,f in enumerate(findings):
        fid = str(f.get("id") or f.get("uid") or i)
        idx_by_id[fid] = i

    # helper to compose the input context
    def input_block(i: int):
        f = findings[i]
        corrs = blob.get("correlations") or []
        # filter to local ones if edges reference id; otherwise just show a few
        local = []
        for c in corrs:
            ids = {str(c.get("src") or c.get("source") or ""),
                   str(c.get("dst") or c.get("target") or "")}
            if str(f.get("id") or f.get("uid") or i) in ids:
                local.append(c)
        txt = render_finding(f) + "\n" + render_correlations(local or corrs[:3])
        return txt

    # Task 1: TRIAGE_SUMMARY
    for i,f in enumerate(findings):
        # find best matching summary text, else skip
        tgt = None
        # direct link
        sid = f.get("summary_id")
        if sid is not None and isinstance(summaries, list) and len(summaries) > 0:
            for s in summaries:
                if str(s.get("id") or s.get("uid")) == str(sid):
                    tgt = s.get("text") or s.get("summary") or None
                    break
        if tgt is None and summaries:
            # heuristic: first summary if only one
            if len(summaries) == 1:
                s = summaries[0]
                tgt = s.get("text") or s.get("summary")
        if tgt:
            rec = TrainRec(
                task="TRIAGE_SUMMARY",
                input_text=input_block(i),
                target_text=tgt,
                label_score=quality_label(blob),
                corr_rho=correlation_strength(blob),
                group_id=str(f.get("id") or i),
            )
            out.append(rec)

    # Task 2: REMEDIATION_ACTIONS
    for i,f in enumerate(findings):
        # link via action_id or fallback to first action
        tgt = None
        aid = f.get("action_id")
        if aid is not None:
            for a in actions:
                if str(a.get("id") or a.get("uid")) == str(aid):
                    tgt = a.get("text") or a.get("remediation") or a.get("steps")
                    break
        if tgt is None and actions:
            if len(actions) == 1:
                a = actions[0]
                tgt = a.get("text") or a.get("remediation") or a.get("steps")
        if tgt:
            rec = TrainRec(
                task="REMEDIATION_ACTIONS",
                input_text=input_block(i),
                target_text=tgt,
                label_score=quality_label(blob),
                corr_rho=correlation_strength(blob),
                group_id=str(f.get("id") or i),
            )
            out.append(rec)

    # Task 3: EXEC_SUMMARY (optional, if reductions exist)
    reds = blob.get("reductions") or []
    if reds:
        # build a compact bundle of top-N findings
        N = min(5, len(findings))
        ctx = []
        for i in range(N):
            ctx.append(render_finding(findings[i]))
        ctx_block = "Bundle context:\n" + "\n---\n".join(ctx)
        tgt = reds[0].get("text") or reds[0].get("summary") or None
        if tgt:
            out.append(TrainRec(
                task="EXEC_SUMMARY",
                input_text=ctx_block,
                target_text=tgt,
                label_score=quality_label(blob),
                corr_rho=correlation_strength(blob),
                group_id="bundle_"+str(reds[0].get("id") or 0),
            ))
    return out
