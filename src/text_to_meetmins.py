# LLM (Ollama) meeting minutes generator.
# Ollama should be running locally (http://localhost:11434)
# to-do: remove bloaty code and reduce unnecessary code.

from __future__ import annotations
import os, json, argparse, sys
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
# Ollama API call. Posts the API request to ollama chat.
def _ollama_chat(base_url: str, model: str, system: str, user: str) -> str:
    r = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"model": model, 
              "messages":[{"role":"system","content":system},
                          {"role":"user","content":user}], 
              "stream": False},
        timeout=1800
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

# clean the LLM output to extract structured data as per the schema
def _parse_json(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n",1)[-1].rsplit("\n",1)[0]
    try:
        return json.loads(s)
    except Exception:
        import re
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except: pass
    return {"summary": s[:800], "key_points": [], "decisions": [], "actions": []}

# Convert LLM output to markdown (JSON to md converter)
def _to_markdown(d: dict, title: str) -> str:
    md = [f"# {title}", "",
          "## Executive Summary", (d.get("summary") or "N/A").strip(), "",
          "## Key Points"]
    kps = d.get("key_points") or []
    md += [f"- {x.strip()}" for x in kps] if kps else ["- N/A"]
    md += ["", "## Decisions"]
    dec = d.get("decisions") or []
    md += [f"- {x.strip()}" for x in dec] if dec else ["- N/A"]
    md += ["", "## Action Items"]
    acts = d.get("actions") or []
    if acts:
        for a in acts:
            task = (a.get("task") or "").strip()
            owner = a.get("owner") or ""
            due = a.get("due") or ""
            ts = a.get("timestamp") or ""
            temp = []
            if owner: temp.append(f"**Owner:** {owner}")
            if due:   temp.append(f"**Due:** {due}")
            if ts:    temp.append(f"**When discussed:** {ts}")
            md.append(f"- {task}" + (f" ({', '.join(temp)})" if temp else ""))
    else:
        md.append("- N/A")
    return "\n".join(md) + "\n"

# LLM meeting minutes generator from transcript text file
def make_minutes_from_text(transcript_text_path: str,
                           out_dir: str = "outputs",
                           model: str = "llama3.1:8b",
                           base_url: str = "http://localhost:11434",
                           title: str = "Meeting Minutes") -> dict:
    if not os.path.exists(transcript_text_path):
        raise FileNotFoundError(transcript_text_path)
    os.makedirs(out_dir, exist_ok=True)
    # create output directory
    base = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(out_dir, f"{base}_minutes_llm"); os.makedirs(run_dir, exist_ok=True)

    with open(transcript_text_path, "r", encoding="utf-8", errors="ignore") as f:
        transcript = f.read()
    # prompt the LLM Chat API
    content = _ollama_chat(base_url, model, SYSTEM_PROMPT, "Transcript:\n\n" + transcript)
    data = _parse_json(content)

    md_path  = os.path.join(run_dir, f"{base}_minutes.md")

    md = _to_markdown(data, title)
    with open(md_path, "w", encoding="utf-8") as f: f.write(md)
    return {"minutes_md": md_path, "dir": run_dir}

# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript_txt")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--base_url", default="http://localhost:11434")
    ap.add_argument("--title", default="Meeting Minutes")
    a = ap.parse_args()
    try:
        res = make_minutes_from_text(a.transcript_txt, a.out_dir, a.model, a.base_url, a.title)
        print("MD:", res["minutes_md"])
    except Exception as e:
        print("ERROR:", e, file=sys.stderr); sys.exit(1)