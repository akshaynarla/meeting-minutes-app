import os
import json
import argparse
from datetime import timedelta
from typing import List, Dict

# pyannote diarization
from pyannote.audio import Pipeline

# Speaker diarization is the process of segmenting audio recordings by speaker labels and 
# aims to answer the question “who spoke when?”. Speaker diarization makes a clear distinction 
# when it is compared with speech recognition. 

# force HH:MM:SS format
def hms(seconds: float) -> str:
    td = timedelta(seconds=round(seconds))
    s = str(td)
    if len(s.split(":")) == 2:
        s = "0:" + s
    return s

# load the transcribed text segments in audio_to_text.py's JSON file
def load_whisper_segments(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])

# pyannote pipeline (this will download models on first run)
def diarize(audio_path: str, hf_token: str = None, num_speakers: int = None):
    if not hf_token:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("Hugging Face token not provided. Set HF_TOKEN env var or pass --hf_token.")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diar = pipe(audio_path, num_speakers=num_speakers) if num_speakers else pipe(audio_path)
    # produce list of (start, end, speaker)
    turns = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
    # sort by start time
    turns.sort(key=lambda x: x["start"])
    return turns

def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

# For each Whisper segment, assign the speaker with the largest time-overlap.
def assign_speakers(whisper_segs: List[Dict], diar_turns: List[Dict]) -> List[Dict]:
    assigned = []
    j = 0
    n = len(diar_turns)
    for seg in whisper_segs:
        ws, we = float(seg["start"]), float(seg["end"])
        best_lbl, best_ov = None, 0.0
        # advance diar pointer to near segment
        while j < n and diar_turns[j]["end"] < ws:
            j += 1
        k = j
        while k < n and diar_turns[k]["start"] < we:
            ov = overlap(ws, we, diar_turns[k]["start"], diar_turns[k]["end"])
            if ov > best_ov:
                best_ov = ov
                best_lbl = diar_turns[k]["speaker"]
            k += 1
        assigned.append({
            "start": ws, "end": we,
            "speaker": best_lbl or "UNKNOWN",
            "text": seg["text"].strip()
        })
    return assigned

# Merge consecutive segments if they have the same speaker and are close in time.
def merge_consecutive(assigned: List[Dict], max_gap: float = 0.5) -> List[Dict]:
    if not assigned:
        return []
    merged = [assigned[0].copy()]
    for seg in assigned[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and seg["start"] - last["end"] <= max_gap:
            # merge
            last["end"] = seg["end"]
            last["text"] = (last["text"] + " " + seg["text"]).strip()
        else:
            merged.append(seg.copy())
    return merged

def write_conversation(merged: List[Dict], out_path: str, include_timestamps: bool = True):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in merged:
            ts = f"[{hms(seg['start'])}]" if include_timestamps else ""
            line = f"{ts} {seg['speaker']}: {seg['text']}".strip()
            f.write(line + "\n")

def write_speaker_map(merged: List[Dict], map_path: str):
    speakers = sorted({seg["speaker"] for seg in merged})
    template = {spk: "" for spk in speakers}
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
    return speakers

def apply_speaker_map(merged: List[Dict], map_path: str):
    if not os.path.exists(map_path):
        return merged
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    renamed = []
    for seg in merged:
        spk = mapping.get(seg["speaker"], seg["speaker"])
        renamed.append({**seg, "speaker": spk if spk else seg["speaker"]})
    return renamed

def main():
    ap = argparse.ArgumentParser(description="Diarize audio and format Whisper transcript as a conversation.")
    ap.add_argument("audio_path", help="Path to audio (mp3/wav/m4a etc.)")
    ap.add_argument("whisper_json", help="Whisper JSON produced by transcribe_whisper.py")
    ap.add_argument("--out", default="conversation.md", help="Output conversation file")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    ap.add_argument("--speakers", type=int, default=None, help="If known, fix number of speakers (e.g., 2)")
    ap.add_argument("--no_timestamps", action="store_true", help="Omit timestamps in the conversation file")
    ap.add_argument("--speaker_map", default="speaker_map.json",
                    help="JSON file to optionally rename speakers after first run")
    args = ap.parse_args()

    print("Loading transribed Whisper segments…")
    whisper_segs = load_whisper_segments(args.whisper_json)

    print("Running diarization in CPU…")
    diar_turns = diarize(args.audio_path, hf_token=args.hf_token, num_speakers=args.speakers)

    print("Assigning speakers to transcribed Whisper segments…")
    assigned = assign_speakers(whisper_segs, diar_turns)

    print("Merging consecutive utterances…")
    merged = merge_consecutive(assigned)

    # write a speaker map template if not present (user can edit to real names and re-run)
    if not os.path.exists(args.speaker_map):
        speakers = write_speaker_map(merged, args.speaker_map)
        print(f"Created speaker map template with speakers: {speakers} -> {args.speaker_map}")
        print("Edit this file to map e.g. 'SPEAKER_00' to 'Akshay', then rerun to apply.")

    # apply map if present
    merged = apply_speaker_map(merged, args.speaker_map)

    print(f"Writing conversation to {args.out} …")
    write_conversation(merged, args.out, include_timestamps=not args.no_timestamps)
    print("Done.")

if __name__ == "__main__":
    main()