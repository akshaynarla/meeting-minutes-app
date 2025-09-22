from __future__ import annotations
import os, re, json
from datetime import datetime
from typing import Dict, List, Optional

# Abstractive summarization (CPU only, model downloads once)
try:
    from transformers import pipeline as hf_pipeline
    HAVE_TRANSFORMERS = True
except Exception:
    HAVE_TRANSFORMERS = False

# Lightweight extractive fallback
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    HAVE_SUMY = True
except Exception:
    HAVE_SUMY = False

ACTION_VERBS = r"(will|to|need to|should|must|plan to|aim to|schedule|send|draft|review|prepare|follow up|create|update|implement|investigate|decide|align|share|present|document)"
DUE_HINT = r"(by|before|on|due|deadline|EOD|end of day|tomorrow|next week|next Monday|Friday|\b\d{4}-\d{2}-\d{2}\b)"
FILLER_WORDS = r"\b(uh|um|you know|like|kind of|sort of)\b"
DECISION_WORDS = r"\b(decided|agree|agreed|decision|conclude|concluded|approved|choose|chose|go with)\b"
DAY_HINT = r"(\b\d{4}-\d{2}-\d{2}\b|\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\b|\b(next week|tomorrow|EOD)\b)"

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _clean_filler(text: str) -> str:
    t = re.sub(FILLER_WORDS, "", text, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Abstractive summarizing involves generating new sentences that capture the essence of the original text.
# An abstractive summarizer presents the material in a logical, well-organized, and grammatically sound form. 
# A summary’s quality can be significantly enhanced by making it more readable or improving its linguistic quality.
def abstractive_summary(text: str, model_name: str = "sshleifer/distilbart-cnn-12-6") -> Optional[str]:
    if not HAVE_TRANSFORMERS:
        return None
    try:
        summarizer = hf_pipeline("summarization", model=model_name)
    except Exception:
        return None

    # chunk into safe windows for model context
    max_chars = 2800
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        last = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
        if last > 0 and len(chunk) > 1200:
            chunk = chunk[:last+1]
        chunks.append(chunk)
        i += len(chunk)

    outs = [summarizer(c, max_length=220, min_length=80, do_sample=False)[0]["summary_text"].strip()
            for c in chunks]
    combined = " ".join(outs)
    if len(outs) > 1:
        final = summarizer(combined, max_length=220, min_length=80, do_sample=False)[0]["summary_text"].strip()
        return final
    return combined

# Extractive summarizing involves picking the most relevant sentences from a document and 
# organizing them systemtically.
# LexRank algorithm, a sentence that is similar to many other sentences of the text 
# has a high probability of being important.
# https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html
def extractive_summary(text: str, sentences: int = 10) -> Optional[str]:
    if not HAVE_SUMY:
        return None
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summ = LexRankSummarizer()
    sents = summ(parser.document, sentences)
    return " ".join(str(s) for s in sents)

def extract_decisions(sentences: List[str]) -> List[str]:
    out = []
    for s in sentences:
        if re.search(DECISION_WORDS, s, re.I):
            out.append(s)
    return out

def extract_actions(sentences: List[str], whisper_json_path: Optional[str]) -> List[Dict]:
    segs = None
    if whisper_json_path and os.path.exists(whisper_json_path):
        with open(whisper_json_path, "r", encoding="utf-8") as f:
            segs = json.load(f).get("segments", [])

    items = []
    for s in sentences:
        if re.search(ACTION_VERBS, s, re.I):
            owner = None
            m = re.match(r"^([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b[:,\- ]", s)
            if m: owner = m.group(1)
            due = None
            if re.search(DUE_HINT, s, re.I):
                mdate = re.search(DAY_HINT, s, re.I)
                if mdate: due = mdate.group(1)
            ts = ""
            if segs:
                for seg in segs:
                    if s.strip() in seg["text"]:
                        ts = f"{seg['start']:.1f}–{seg['end']:.1f}s"
                        break
            items.append({"task": s, "owner": owner or "", "due": due or "", "timestamp": ts})
    return items

def write_minutes_md(
    path_md: str,
    title: str,
    summary: str,
    key_points: List[str],
    decisions: List[str],
    actions: List[Dict],
):
    lines = [f"# {title}", "", "## Executive Summary", summary or "N/A", "", "## Key Points"]
    if key_points:
        lines += [f"- {kp}" for kp in key_points]
    else:
        lines.append("- N/A")
    lines += ["", "## Decisions"]
    if decisions:
        lines += [f"- {d}" for d in decisions]
    else:
        lines.append("- N/A")
    lines += ["", "## Action Items"]
    if actions:
        for a in actions:
            meta = []
            if a["owner"]: meta.append(f"**Owner:** {a['owner']}")
            if a["due"]:   meta.append(f"**Due:** {a['due']}")
            if a["timestamp"]: meta.append(f"**When discussed:** {a['timestamp']}")
            suffix = f" ({', '.join(meta)})" if meta else ""
            lines.append(f"- {a['task']}{suffix}")
    else:
        lines.append("- N/A")
    _write(path_md, "\n".join(lines) + "\n")

def write_actions_csv(path_csv: str, actions: List[Dict]):
    if not actions:
        return
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["task","owner","due","timestamp"])
        w.writeheader()
        w.writerows(actions)

def make_minutes_from_text(
    transcript_text_path: str,
    whisper_json_path: Optional[str] = None,
    out_dir: str = "outputs",
    summary_mode: str = "auto",
    abstractive_model: str = "sshleifer/distilbart-cnn-12-6",
    key_points_n: int = 8,
) -> Dict[str, str]:
    if not os.path.exists(transcript_text_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_text_path}")

    base = os.path.splitext(os.path.basename(transcript_text_path))[0]
    run_dir = os.path.join(out_dir, f"{base}_minutes")
    os.makedirs(run_dir, exist_ok=True)

    raw = _read(transcript_text_path)
    cleaned = _clean_filler(raw)
    sentences = _split_sentences(cleaned)

    summary = None
    if summary_mode in ("auto", "abstractive"):
        summary = abstractive_summary(cleaned, model_name=abstractive_model)
    if not summary and summary_mode in ("auto", "extractive"):
        summary = extractive_summary(cleaned, sentences=10)
    if not summary:
        summary = " ".join(sentences[:min(8, len(sentences))])

    key_points = sentences[:max(3, min(key_points_n, len(sentences)))]
    decisions = extract_decisions(sentences)
    actions = extract_actions(sentences, whisper_json_path)

    md_path = os.path.join(run_dir, f"{base}_minutes.md")
    csv_path = os.path.join(run_dir, f"{base}_actions.csv")

    title = f"Meeting Minutes – {datetime.now().strftime('%Y-%m-%d')}"
    write_minutes_md(md_path, title, summary, key_points, decisions, actions)
    write_actions_csv(csv_path, actions)

    return {"minutes_md": md_path, "actions_csv": csv_path, "dir": run_dir}
