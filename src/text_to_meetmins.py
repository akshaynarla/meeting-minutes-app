# LLM (Ollama) meeting minutes generator.
# Ollama should be running locally (http://localhost:11434)
# to-do: remove bloaty code and reduce unnecessary code.

from __future__ import annotations
import os, re, json, csv, argparse, sys
from typing import Dict, List
import requests
from datetime import datetime

SYSTEM_PROMPT = """You are an expert meeting scribe.
Given a meeting transcript, produce STRICT JSON with:
1) "summary": a faithful, concise 5–8 sentence executive summary
2) "key_points": up to 10 bullet-level points
3) "decisions": only finalized decisions (no plans)
4) "actions": list of { "task", "owner", "due", "timestamp" }
   - "owner": person responsible if known, else ""
   - "due": explicit due (e.g., "2025-09-23", "next week", "Friday", "EOD"), else ""
   - "timestamp": if transcript lines contain [HH:MM:SS] near where the action was discussed, include one; else ""
Rules:
- Be faithful. Do NOT invent names, dates, or outcomes.
- Use short, clear sentences. No fluff.
Return ONLY valid JSON (no markdown fences, no commentary).
Schema:
{
  "summary": "string",
  "key_points": ["string", ...],
  "decisions": ["string", ...],
  "actions": [{"task":"string","owner":"string","due":"string","timestamp":"string"}, ...]
}
"""

def _chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts, i = [], 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        last = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "), chunk.rfind("\n"))
        if last > 1000:
            chunk = chunk[:last+1]
        parts.append(chunk)
        i += len(chunk)
    return parts

def _call_ollama(messages: List[Dict[str, str]], model: str, base_url: str) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=1800)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

def _extract_json(text: str) -> dict:
    s = text.strip()
    s = re.sub(r"^```(json)?\s*|\s*```$", "", s, flags=re.I | re.M).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    # last resort fallback
    return {"summary": s[:1000], "key_points": [], "decisions": [], "actions": []}

def _combine_minutes(parts: List[dict]) -> dict:
    out = {"summary": "", "key_points": [], "decisions": [], "actions": []}
    for p in parts:
        if not isinstance(p, dict): 
            continue
        if p.get("summary"):
            out["summary"] += (" " if out["summary"] else "") + str(p["summary"]).strip()
        out["key_points"].extend([x for x in (p.get("key_points") or []) if isinstance(x, str)])
        out["decisions"].extend([x for x in (p.get("decisions") or []) if isinstance(x, str)])
        for a in p.get("actions") or []:
            if isinstance(a, dict) and a.get("task"):
                out["actions"].append({
                    "task": str(a.get("task","")).strip(),
                    "owner": str(a.get("owner","")).strip(),
                    "due": str(a.get("due","")).strip(),
                    "timestamp": str(a.get("timestamp","")).strip(),
                })
    def _dedupe(seq):
        seen, res = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x); res.append(x)
        return res
    out["key_points"] = _dedupe(out["key_points"])[:10]
    out["decisions"] = _dedupe(out["decisions"])
    seen = set(); uniq=[]
    for a in out["actions"]:
        key = (a["task"], a["owner"], a["due"], a["timestamp"])
        if key not in seen:
            seen.add(key); uniq.append(a)
    out["actions"] = uniq
    return out

def _minutes_json_to_markdown(d: dict, title: str) -> str:
    lines = [f"# {title}", "", "## Executive Summary", d.get("summary","").strip() or "N/A", "", "## Key Points"]
    kps = d.get("key_points") or []
    lines += [f"- {kp.strip()}" for kp in kps] if kps else ["- N/A"]
    lines += ["", "## Decisions"]
    dec = d.get("decisions") or []
    lines += [f"- {x.strip()}" for x in dec] if dec else ["- N/A"]
    lines += ["", "## Action Items"]
    acts = d.get("actions") or []
    if acts:
        for a in acts:
            meta=[]
            if a.get("owner"): meta.append(f"**Owner:** {a['owner']}")
            if a.get("due"):   meta.append(f"**Due:** {a['due']}")
            if a.get("timestamp"): meta.append(f"**When discussed:** {a['timestamp']}")
            suffix = f" ({', '.join(meta)})" if meta else ""
            lines.append(f"- {a['task']}{suffix}")
    else:
        lines.append("- N/A")
    return "\n".join(lines) + "\n"

def make_minutes_from_text(
    transcript_text_path: str,
    out_dir: str = "outputs",
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    title: str = "Meeting Minutes",
) -> Dict[str, str]:
    if not os.path.exists(transcript_text_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_text_path}")
    os.makedirs(out_dir, exist_ok=True)

    base = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(out_dir, f"{base}_minutes_llm")
    os.makedirs(run_dir, exist_ok=True)

    with open(transcript_text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = _chunk_text(full_text, max_chars=12000)
    partials = []
    for idx, chunk in enumerate(chunks, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript chunk {idx}/{len(chunks)}:\n\n{chunk}"}
        ]
        content = _call_ollama(messages, model=model, base_url=base_url)
        partials.append(_extract_json(content))

    merged = _combine_minutes(partials)

    md_path  = os.path.join(run_dir, f"{base}_minutes.md")
    txt_path = os.path.join(run_dir, f"{base}_minutes.txt")

    md = _minutes_json_to_markdown(merged, title=title)
    with open(md_path,  "w", encoding="utf-8") as f: f.write(md)
    with open(txt_path, "w", encoding="utf-8") as f:
        # plain-text version (no markdown headers)
        f.write(f"{title}\n\nEXECUTIVE SUMMARY\n{merged.get('summary','').strip()}\n\nKEY POINTS\n")
        for kp in (merged.get("key_points") or []): f.write(f"- {kp.strip()}\n")
        f.write("\nDECISIONS\n")
        for d in (merged.get("decisions") or []): f.write(f"- {d.strip()}\n")
        f.write("\nACTION ITEMS\n")
        for a in (merged.get("actions") or []):
            meta = []
            if a.get("owner"): meta.append(f"Owner: {a['owner']}")
            if a.get("due"):   meta.append(f"Due: {a['due']}")
            if a.get("timestamp"): meta.append(f"When: {a['timestamp']}")
            suffix = f" ({', '.join(meta)})" if meta else ""
            f.write(f"- {a.get('task','').strip()}{suffix}\n")

    if merged.get("actions"):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["task","owner","due","timestamp"])
            w.writeheader()
            for a in merged["actions"]:
                w.writerow(a)
    else:
        csv_path = ""

    return {"minutes_md": md_path, "minutes_txt": txt_path, "actions_csv": csv_path, "dir": run_dir}

# CLI for quick tests
def _parse_args():
    ap = argparse.ArgumentParser(description="LLM-only minutes (Ollama)")
    ap.add_argument("transcript_txt", help="Path to transcript .txt or conversation .md")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--base_url", default="http://localhost:11434")
    ap.add_argument("--title", default="Meeting Minutes")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    try:
        r = make_minutes_from_text(
            transcript_text_path=args.transcript_txt,
            out_dir=args.out_dir,
            model=args.model,
            base_url=args.base_url,
            title=args.title,
        )
        print("✅ Minutes MD:", r["minutes_md"])
        print("✅ Minutes TXT:", r["minutes_txt"])
        if r["actions_csv"]:
            print("✅ Actions CSV:", r["actions_csv"])
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)