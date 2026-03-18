# src/backend/backend_llm.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse

import requests
from .backend_rag import DocumentRAG, RAG_AVAILABLE

_CONVERSATION_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*\[(\d\d:\d\d:\d\d)\]:\s*(.*)$")

SYSTEM_PROMPT = """You are a professional meeting minute-taker.
Write the meeting minutes in CLEAR, formatted Markdown.

Required Output Structure:
## 📝 Executive Summary
[Brief 5-8 sentence summary here]

## 🔑 Key Points
- [Key point 1]
- [Key point 2]

## 🤝 Decisions Made
- [Decision 1]
- [Decision 2]

## ✅ Action Items
| Task | Owner | Due |
|---|---|---|
| [Task] | [Owner] | [Date] |

Guidelines:
- Be factual and concise.
- Use ONLY the provided context. If unknown, use "TBD".
- If a section has no content, put "- (None)".
- Return exactly the Markdown structure above. Do not surround it with ```markdown blocks.
"""


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
    timeout: Optional[int] = 600,
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
        if resp.status_code != 200:
            try:
                err_text = resp.json().get("error", resp.text)
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {err_text}")
            except ValueError:
                resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "") or ""

    # Streaming response: NDJSON
    content = ""
    with requests.post(url, json=payload, timeout=timeout, stream=True) as resp:
        if resp.status_code != 200:
            try:
                err_text = resp.json().get("error", resp.text)
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {err_text}")
            except ValueError:
                resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(data["error"])
            
            msg = data.get("message") or {}
            token = msg.get("content", "") or ""
            think_token = msg.get("thinking", "") or ""
            
            if think_token:
                print(think_token, end="", flush=True)
            if token:
                print(token, end="", flush=True)
                
            content += token
            if data.get("done"):
                print() # Add a finishing newline
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

    chunks = _chunk_by_turns(transcript_text, max_chars=6500)
    stream = os.getenv("OLLAMA_STREAM", "0").strip().lower() in {"1", "true", "yes"}

    if len(chunks) <= 1 or not RAG_AVAILABLE:
        # Short transcript or RAG disabled -> direct fallback
        final_input = transcript_text
    else:
        # RAG Logic Phase
        topics_list = []
        for idx, chunk in enumerate(chunks, start=1):
            messages = [
                {
                    "role": "system",
                    "content": "Identify up to 3 most important topics discussed in this chunk. Return ONLY a comma-separated list of short topic names without numbers or explanations."
                },
                {"role": "user", "content": f"Chunk:\n\n{chunk}"},
            ]
            
            content = _ollama_chat(
                base_url, model, messages, temperature=0.1, timeout=None, stream=stream
            )
            # Extracted distinct topic phrases
            topics = [t.strip().strip('-*').replace('\n','') for t in content.split(',') if t.strip()]
            topics_list.extend(topics)
            
        unique_topics = []
        for t in topics_list:
            if t.lower() not in [ut.lower() for ut in unique_topics]:
                unique_topics.append(t)
        unique_topics = unique_topics[:6] # Limit topics to avoid gigantic payloads
        
        # Build vector retrieval index via DocumentRAG
        rag = DocumentRAG()
        rag_chunks = _chunk_by_turns(transcript_text, max_chars=1200)
        rag.build_index(rag_chunks)
        
        # Search the Exact Quotes
        context_blocks = []
        for topic in unique_topics:
            retrieved = rag.retrieve(f"Find discussions, decisions, and action items about: {topic}", k=3)
            context_blocks.append(f"### Topic: {topic}\n" + "\n".join(retrieved))
            
        final_input = (
            "Here is the exact retrieved context from the meeting organized by topic. "
            "Use these EXACT facts to generate your final minutes, decisions, and action items.\n\n"
            + "\n\n".join(context_blocks)
        )

    # Final minutes (structured output)
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Meeting transcription and details:\n\n{final_input}"},
    ]

    final_content = _ollama_chat(
        base_url,
        model,
        final_messages,
        temperature=0.2,
        timeout=None,
        stream=stream,
    )

    # Clean markdown fences if the model outputted them
    final_content = final_content.strip()
    if final_content.startswith("```markdown"):
        final_content = final_content.replace("```markdown\n", "", 1)
    if final_content.startswith("```"):
        final_content = final_content.replace("```\n", "", 1)
    if final_content.endswith("```"):
        final_content = final_content[:-3]

    md_content = f"# {title}\n\n**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n" + final_content.strip() + "\n"

    out_dir = os.path.dirname(transcript_path)
    out_path = os.path.join(out_dir, "meeting_minutes.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return out_path