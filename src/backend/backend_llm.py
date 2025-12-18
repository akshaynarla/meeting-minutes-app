# src/backend/backend_llm.py
import requests
import json
import re
import os
from datetime import datetime
from typing import Dict, Any

SYSTEM_PROMPT = """You are a professional minute-taker. 
You must output a VALID JSON object based on the transcript provided.
Do not add any preamble or markdown formatting like ```json. Just the raw JSON.

Structure your JSON exactly like this:
{
    "summary": "A concise executive summary (5-8 sentences).",
    "key_points": ["Bullet point 1", "Bullet point 2", ...],
    "decisions": ["Decision 1", "Decision 2", ...],
    "action_items": [
        {"task": "What needs to be done", "owner": "Who is responsible", "due_date": "When (or TBD)"}
    ]
}
"""

def _repair_json(bad_json: str) -> Dict[str, Any]:
    clean = bad_json.replace("```json", "").replace("```", "").strip()
    m = re.search(r"(\{.*\})", clean, re.DOTALL)
    if m:
        clean = m.group(1)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"summary": "Error parsing JSON from LLM output.",
                "key_points": [], "decisions": [], "action_items": []}

def generate_minutes(
    transcript_path: str,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    title: str = "Meeting Minutes"
) -> str:
    if not os.path.exists(transcript_path):
        raise FileNotFoundError("Transcript file not found.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    if len(transcript_text) > 50000:
        transcript_text = transcript_text[:50000] + "...[TRUNCATED]"

    # Chunking
    CHUNK_SIZE = 4000          # characters per chunk
    CHUNK_NUM_PREDICT = 256    # how long each chunk summary can be
    FINAL_NUM_PREDICT = 512    # how long the final JSON answer can be

    if len(transcript_text) > CHUNK_SIZE:
        chunks = [
            transcript_text[i:i + CHUNK_SIZE]
            for i in range(0, len(transcript_text), CHUNK_SIZE)
        ]

        chunk_summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant summarising a meeting transcript chunk. "
                        "Summarise ONLY this chunk, focusing on key points, decisions, and "
                        "action items. Keep it concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is transcript chunk {idx} of {len(chunks)}:\n\n"
                        f"{chunk}\n\n"
                        "Summarise this chunk only."
                    ),
                },
            ]

            chunk_payload = {
                "model": model,
                "messages": chunk_messages,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": CHUNK_NUM_PREDICT,
                },
            }

            resp = requests.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=chunk_payload,
                timeout=600,
            )
            resp.raise_for_status()
            chunk_text = resp.json()["message"]["content"].strip()
            chunk_summaries.append(chunk_text)

        # Replace long raw transcript with combined chunk summaries
        transcript_text = "\n\n".join(chunk_summaries)
    # ---------- END CHUNKING SECTION ----------

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is the transcript:\n\n{transcript_text}"}
        ],
        "stream": False,
        "options": {"temperature": 0.2}
    }

    resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=600)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    data = _repair_json(content)

    md = [f"# {title}", "", f"**Date:** {datetime.now().strftime('%Y-%m-%d')}", ""]
    md += ["## 📝 Executive Summary", data.get("summary", "N/A"), ""]
    md += ["## 🔑 Key Points"] + [f"- {kp}" for kp in data.get("key_points", [])] + [""]
    md += ["## 🤝 Decisions Made"] + [f"- {d}" for d in data.get("decisions", [])] + [""]
    md += ["## ✅ Action Items"]
    actions = data.get("action_items", [])
    if actions:
        md += ["", "| Task | Owner | Due |", "|---|---|---|"]
        for a in actions:
            md += [f"| {a.get('task','')} | {a.get('owner','')} | {a.get('due_date','')} |"]
    else:
        md += ["No specific action items detected."]
    md_content = "\n".join(md) + "\n"

    out_dir = os.path.dirname(transcript_path)
    out_path = os.path.join(out_dir, "meeting_minutes.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return out_path
