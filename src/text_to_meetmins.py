import argparse, re, os, csv, math
from datetime import datetime
from pathlib import Path

# Abstractive summarization (downloads a model once)
USE_TRANSFORMERS = True
try:
    from transformers import pipeline, AutoTokenizer
except Exception:
    USE_TRANSFORMERS = False

# Lightweight extractive fallback (very small, CPU-only)
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    HAVE_SUMY = True
except Exception:
    HAVE_SUMY = False

# Patterns for action item extraction
ACTION_VERBS = r"(will|to|need to|should|must|plan to|aim to|schedule|send|draft|review|prepare|follow up|create|update|implement|investigate|decide|align|share|present|document)"
DUE_HINT = r"(by|before|on|due|deadline|EOD|end of day|tomorrow|next week|next Monday|Friday|[0-9]{4}-[0-9]{2}-[0-9]{2})"
FILLER_WORDS = r"\b(uh|um|like|kind of|sort of)\b"
DECISION_WORDS = r"\b(decided|agree|agreed|decision|conclude|concluded|approved|choose|chose|go with)\b"

def read_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

# Remove whitespaces and strip filler tokens
def clean_text(t: str) -> str:
    t = re.sub(FILLER_WORDS, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# simple splitter to avoid large deps
def split_into_sentences(t: str):
    t = t.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [s.strip() for s in parts if s.strip()]

def abstractive_summary(text: str, model_name="sshleifer/distilbart-cnn-12-6",
                        max_chunk_chars=2800, min_len=80, max_len=220):
    if not USE_TRANSFORMERS:
        return None
    try:
        summarizer = pipeline("summarization", model=model_name)
    except Exception:
        return None
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chunk_chars]
        # try to not split mid-sentence
        last = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("! "))
        if last > 0 and len(chunk) > 1200:
            chunk = chunk[:last+1]
        chunks.append(chunk)
        i += len(chunk)
    outs = []
    for c in chunks:
        out = summarizer(c, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        outs.append(out.strip())
    # If multiple chunks, summarize the summaries
    combined = " ".join(outs)
    if len(outs) > 1:
        final = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        return final.strip()
    return combined.strip()

def extractive_summary(text: str, sentences=10):
    if not HAVE_SUMY:
        return None
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join(str(s) for s in summary)

def extract_decisions(sentences):
    decisions = []
    for s in sentences:
        if re.search(DECISION_WORDS, s, re.I):
            decisions.append(s)
    return decisions

def extract_actions(sentences, segments=None):
    items = []
    for s in sentences:
        if re.search(ACTION_VERBS, s, re.I):
            owner = None
            # naive owner guess: leading proper noun or "I/We/John/…"
            m = re.match(r"^([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b[:,\- ]", s)
            if m: owner = m.group(1)
            due = None
            if re.search(DUE_HINT, s, re.I):
                # try to pull a simple date-like token
                mdate = re.search(r"(\b\d{4}-\d{2}-\d{2}\b|\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\b|\b(next week|tomorrow|EOD)\b)", s, re.I)
                if mdate: due = mdate.group(1)
            ts = None
            if segments:
                # crude mapping: find a segment whose text contains this sentence
                for seg in segments:
                    if s.strip() in seg["text"]:
                        ts = f"{seg['start']:.1f}–{seg['end']:.1f}s"
                        break
            items.append({"task": s, "owner": owner or "", "due": due or "", "timestamp": ts or ""})
    return items

def load_segments_json(path):
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("segments", [])
    except Exception:
        return None

def write_markdown(path, title, summary, key_points, decisions, actions):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("## Executive Summary\n")
        f.write((summary or "N/A").strip() + "\n\n")
        f.write("## Key Points\n")
        if key_points:
            for kp in key_points:
                f.write(f"- {kp}\n")
        else:
            f.write("- N/A\n")
        f.write("\n## Decisions\n")
        if decisions:
            for d in decisions:
                f.write(f"- {d}\n")
        else:
            f.write("- N/A\n")
        f.write("\n## Action Items\n")
        if actions:
            for a in actions:
                line = f"- {a['task']}"
                meta = []
                if a["owner"]: meta.append(f"**Owner:** {a['owner']}")
                if a["due"]:   meta.append(f"**Due:** {a['due']}")
                if a["timestamp"]: meta.append(f"**When discussed:** {a['timestamp']}")
                if meta: line += "  (" + ", ".join(meta) + ")"
                f.write(line + "\n")
        else:
            f.write("- N/A\n")

def write_actions_csv(path, actions):
    if not actions: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task","owner","due","timestamp"])
        w.writeheader()
        w.writerows(actions)

def meeting_minutes(args):
    # Read and preprocess transcript
    raw = read_text(args.transcript_txt)
    text = clean_text(raw)
    sents = split_into_sentences(text)

    # Summary
    summary = None
    if args.summary_mode in ("auto","abstractive"):
        summary = abstractive_summary(text, model_name=args.abstractive_model)
    if not summary and args.summary_mode in ("auto","extractive"):
        summary = extractive_summary(text, sentences=10)

    # Key points: pick top-N representative sentences (very simple)
    kp = sents[:max(3, min(args.key_points, len(sents)))]

    # Decisions + Actions
    segments = load_segments_json(args.segments_json) if args.segments_json else None
    decisions = extract_decisions(sents)
    actions = extract_actions(sents, segments)

    # Write outputs
    title = f"Meeting Minutes – {datetime.now().strftime('%Y-%m-%d')}"
    write_markdown(args.out_md, title, summary, kp, decisions, actions)
    write_actions_csv(args.out_csv, actions)

    print("Minutes written:", os.path.abspath(args.out_md))
    if actions:
        print("Action items CSV:", os.path.abspath(args.out_csv))