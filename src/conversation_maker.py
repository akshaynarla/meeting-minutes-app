from __future__ import annotations
import os, json, re
from datetime import timedelta
from typing import Dict, List, Optional

# pyannote pulls huggingface_hub and also just CPU. You'll need a HF read token.
from pyannote.audio import Pipeline as PyannotePipeline

def _hms(seconds: float) -> str:
    td = timedelta(seconds=round(seconds))
    s = str(td)
    if len(s.split(":")) == 2:
        s = "0:" + s
    return s

def run_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    num_speakers: Optional[int] = None,
    model_id: str = "pyannote/speaker-diarization-3.1",
) -> List[Dict]:
    """
    Returns diarization turns: [{start, end, speaker}, ...]
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Hugging Face 'read' token required (set HF_TOKEN or pass to run_diarization).")

    pipe = PyannotePipeline.from_pretrained(model_id, use_auth_token=hf_token)
    diar = pipe(audio_path, num_speakers=num_speakers) if num_speakers else pipe(audio_path)

    turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
    turns.sort(key=lambda x: x["start"])
    return turns

def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def assign_speakers_to_whisper_segments(
    whisper_json_path: str,
    diar_turns: List[Dict],
    merge_gap_sec: float = 0.5,
) -> List[Dict]:
    """
    Reads Whisper JSON, assigns speaker label to each segment by max overlap,
    and merges consecutive same-speaker segments (<= merge_gap_sec apart).
    Returns [{start, end, speaker, text}, ...]
    """
    if not os.path.exists(whisper_json_path):
        raise FileNotFoundError(f"Whisper JSON not found: {whisper_json_path}")

    with open(whisper_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    wsegs = data.get("segments", [])

    assigned = []
    j, n = 0, len(diar_turns)
    for seg in wsegs:
        ws, we = float(seg["start"]), float(seg["end"])
        best_lbl, best_ov = None, 0.0
        while j < n and diar_turns[j]["end"] < ws:
            j += 1
        k = j
        while k < n and diar_turns[k]["start"] < we:
            ov = _overlap(ws, we, diar_turns[k]["start"], diar_turns[k]["end"])
            if ov > best_ov:
                best_ov, best_lbl = ov, diar_turns[k]["speaker"]
            k += 1
        assigned.append({"start": ws, "end": we, "speaker": best_lbl or "UNKNOWN", "text": seg["text"].strip()})

    # merge consecutive same-speaker segments
    merged: List[Dict] = []
    for seg in assigned:
        if merged and seg["speaker"] == merged[-1]["speaker"] and seg["start"] - merged[-1]["end"] <= merge_gap_sec:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
        else:
            merged.append(seg)
    return merged

def write_conversation_markdown(
    segments: List[Dict],
    out_path: str,
    timestamps: bool = True,
):
    lines = []
    for s in segments:
        ts = f"[{_hms(s['start'])}] " if timestamps else ""
        lines.append(f"{ts}{s['speaker']}: {s['text']}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def create_or_load_speaker_map(map_path: str, segments: List[Dict]) -> Dict[str, str]:
    """
    If map file exists, load and return. Else create template and return it.
    Template maps 'SPEAKER_00' to "" etc., ready for user edits.
    """
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    speakers = sorted({s["speaker"] for s in segments})
    mapping = {spk: "" for spk in speakers}
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return mapping

def apply_speaker_map(segments: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
    out = []
    for s in segments:
        spk = mapping.get(s["speaker"], s["speaker"])
        out.append({**s, "speaker": spk if spk else s["speaker"]})
    return out