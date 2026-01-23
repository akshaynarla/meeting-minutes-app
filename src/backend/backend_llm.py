# src/backend/backend_llm.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse

import requests

_CONVERSATION_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d\d:\d\d:\d\d)\]:\s*(.*)$")

SYSTEM_PROMPT = """You are a professional meeting minute-taker.

Return a VALID JSON object ONLY (no markdown fences, no preamble).
Your JSON MUST match the provided schema.

Guidelines:
- Be factual and concise.
- Prefer specific, actionable items.
- If something is unknown, use "TBD".
"""


MINUTES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Executive summary (5-8 sentences)."},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due_date": {"type": "string"},
                },
                "required": ["task", "owner", "due_date"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "key_points", "decisions", "action_items"],
    "additionalProperties": False,
}

CHUNK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "key_points": {"type": "array", "items": {"type": "string"}},
        "decisions": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": "string"},
                    "due_date": {"type": "string"},
                },
                "required": ["task", "owner", "due_date"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["key_points", "decisions", "action_items"],
    "additionalProperties": False,
}


def _repair_json(bad_json: str) -> Dict[str, Any]:
    """For improved output from LLMs: Best-effort JSON extraction when the model ignores structured output."""
    clean = bad_json.replace("```json", "").replace("```", "").strip()
    m = re.search(r"(\{.*\})", clean, re.DOTALL)
    if m:
        clean = m.group(1)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "summary": "Error parsing JSON from LLM output.",
            "key_points": [],
            "decisions": [],
            "action_items": [],
        }


def _is_localhost_url(base_url: str) -> bool:
    p = urlparse(base_url)
    host = (p.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _enforce_local_only(base_url: str) -> None:
    """Refuse non-local Ollama endpoints unless explicitly allowed."""
    allow_remote = os.getenv("OLLAMA_ALLOW_REMOTE", "0").strip().lower() in {"1", "true", "yes"}
    if allow_remote:
        return
    if not _is_localhost_url(base_url):
        raise ValueError(
            "Refusing to connect to a non-local Ollama endpoint. "
            "Set environment variable OLLAMA_ALLOW_REMOTE=1 to override."
        )


def _ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    num_predict: Optional[int] = None,
    timeout: int = 600,
    stream: bool = False,
) -> str:
    """Call Ollama /api/chat and return the assistant message content as a string."""
    _enforce_local_only(base_url)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = int(num_predict)
    if schema is not None:
        # Ollama structured outputs: pass JSON schema via 'format'
        payload["format"] = schema

    url = f"{base_url.rstrip('/')}/api/chat"

    if not stream:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "") or ""

    # Streaming response: NDJSON
    content = ""
    with requests.post(url, json=payload, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(data["error"])
            token = (data.get("message") or {}).get("content", "") or ""
            content += token
            if data.get("done"):
                break
    return content


def _parse_conversation_md(text: str) -> List[Tuple[str, str, str]]:
    """Parse conversation.md lines => list of (speaker, hh:mm:ss, text)."""
    turns: List[Tuple[str, str, str]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = _CONVERSATION_LINE_RE.match(raw)
        if not m:
            continue
        speaker, ts, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if body:
            turns.append((speaker, ts, body))
    return turns

# vibe-coded with co-pilot for optimizing chunking
def _chunk_by_turns(transcript_text: str, max_chars: int = 6500) -> List[str]:
    """Chunk a transcript. If it looks like conversation.md, chunk by turns (speaker-wise for example);
      else chunk by paragraphs."""
    turns = _parse_conversation_md(transcript_text)
    if turns:
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0
        for speaker, ts, body in turns:
            line = f"**{speaker}** [{ts}]: {body}"
            if cur and cur_len + len(line) + 1 > max_chars:
                chunks.append("\n".join(cur))
                cur = [line]
                cur_len = len(line)
            else:
                cur.append(line)
                cur_len += len(line) + 1
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    # Fallback: chunk by paragraphs
    paras = [p.strip() for p in re.split(r"\n\s*\n", transcript_text) if p.strip()]
    if not paras:
        return [transcript_text.strip()] if transcript_text.strip() else []

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in paras:
        if cur and cur_len + len(p) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _merge_chunk_notes(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    ''' Merge the important extracted points from each chunk
      to provide consolidated meeting minutes'''
    key_points: List[str] = []
    decisions: List[str] = []
    action_items: List[Dict[str, str]] = []

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    seen_kp, seen_dec, seen_ai = set(), set(), set()

    for n in notes:
        for kp in n.get("key_points", []) or []:
            nk = _norm(kp)
            if nk and nk not in seen_kp:
                seen_kp.add(nk)
                key_points.append(kp.strip())

        for d in n.get("decisions", []) or []:
            nd = _norm(d)
            if nd and nd not in seen_dec:
                seen_dec.add(nd)
                decisions.append(d.strip())

        for a in n.get("action_items", []) or []:
            task = str(a.get("task", "")).strip()
            owner = str(a.get("owner", "")).strip() or "TBD"
            due = str(a.get("due_date", "")).strip() or "TBD"
            if not task:
                continue
            sig = _norm(task)
            if sig in seen_ai:
                continue
            seen_ai.add(sig)
            action_items.append({"task": task, "owner": owner, "due_date": due})

    return {"key_points": key_points, "decisions": decisions, "action_items": action_items}


def generate_minutes(
    transcript_path: str,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    title: str = "Meeting Minutes"
) -> str:
    """Generate meeting minutes from a transcript file.

    Transcript can be:
      - conversation.md produced by the whisperx framework, or
      - plain text transcript

    Output:
      - meeting_minutes.md in the transcript directory
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError("Transcript file not found.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read().strip()

    if not transcript_text:
        raise ValueError("Transcript is empty.")

    # Chunk transcript without truncating meeting content, while retaining speaker turns.
    # Returns a list with chunks
    chunks = _chunk_by_turns(transcript_text, max_chars=6500)

    # Summarize chunks into structured notes (hierarchical summarization).
    stream = os.getenv("OLLAMA_STREAM", "0").strip().lower() in {"1", "true", "yes"}
    chunk_notes: List[Dict[str, Any]] = []
    if len(chunks) > 1:
        # More than 1 chunk i.e. longer audio needs multiple round of processing 
        # to support the context window of the LLM and not lose any data
        for idx, chunk in enumerate(chunks, start=1):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract ONLY from the provided chunk. "
                        "Return key points, decisions, and action items. "
                        "If unknown, use 'TBD'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Transcript chunk {idx} of {len(chunks)}:\n\n{chunk}",
                },
            ]

            # First attempt: structured output
            content = _ollama_chat(
                base_url,
                model,
                messages,
                schema=CHUNK_SCHEMA,
                temperature=0.2,
                num_predict=256,
                timeout=600,
                stream=stream,
            )

            note = _repair_json(content)
            # Ensure required keys exist even if the model ignored schema
            note.setdefault("key_points", [])
            note.setdefault("decisions", [])
            note.setdefault("action_items", [])
            chunk_notes.append(note)

        # merge the minutes from each chunk into one variable
        merged = _merge_chunk_notes(chunk_notes)

        final_input = (
            "Here are aggregated notes extracted from the full transcript. "
            "Use these to write final minutes.\n\n"
            + json.dumps(merged, ensure_ascii=False, indent=2)
        )
    else:
        # Short transcript: send directly for best fidelity.
        final_input = transcript_text

    # Final minutes (structured output)
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Meeting transcript or notes:\n\n{final_input}"},
    ]

    final_content = _ollama_chat(
        base_url,
        model,
        final_messages,
        schema=MINUTES_SCHEMA,
        temperature=0.2,
        num_predict=512,
        timeout=900,
        stream=stream,
    )

    # format the output from ollama/LLMs for better output
    data = _repair_json(final_content)
    data.setdefault("summary", "N/A")
    data.setdefault("key_points", [])
    data.setdefault("decisions", [])
    data.setdefault("action_items", [])

    # vibe-coded to reduce LOC. Looks rather complicated, 
    # but more or less is List Comprehension
    md: List[str] = [
        f"# {title}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 📝 Executive Summary",
        str(data.get("summary", "N/A")).strip(),
        "",
        "## 🔑 Key Points",
    ]
    md += [f"- {kp}" for kp in (data.get("key_points") or [])] or ["- (None)"]
    md += ["", "## 🤝 Decisions Made"]
    md += [f"- {d}" for d in (data.get("decisions") or [])] or ["- (None)"]
    md += ["", "## ✅ Action Items"]

    # extract action items from the meeting
    actions = data.get("action_items") or []
    if actions:
        md += ["", "| Task | Owner | Due |", "|---|---|---|"]
        for a in actions:
            md += [f"| {a.get('task','').strip()} | {a.get('owner','').strip()} | {a.get('due_date','').strip()} |"]
    else:
        md += ["- (None)"]

    md_content = "\n".join(md).strip() + "\n"

    out_dir = os.path.dirname(transcript_path)
    out_path = os.path.join(out_dir, "meeting_minutes.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return out_path