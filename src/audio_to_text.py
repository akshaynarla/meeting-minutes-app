from __future__ import annotations
import os, json
from typing import Optional, Dict
import whisper
from whisper.utils import get_writer

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def transcribe_whisper(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    translate: bool = False,
    output_dir: str = "outputs"
) -> Dict[str, str]:
    """
    CPU-only transcription with Whisper.
    Returns dict with paths: text, json, srt, vtt, dir.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    _ensure_dir(output_dir)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    run_dir = _ensure_dir(os.path.join(output_dir, f"{base}_transcript"))

    model = whisper.load_model(model_size, device="cpu")
    result = model.transcribe(
        audio_path,
        fp16=False,  # CPU
        language=language,
        task="translate" if translate else "transcribe",
        verbose=False,
    )

    txt = os.path.join(run_dir, f"{base}.txt")
    jsn = os.path.join(run_dir, f"{base}.json")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(result["text"].strip() + "\n")
    with open(jsn, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Sidecar subtitle files
    for ext in ["srt", "vtt", "tsv"]:
        writer = get_writer(ext, run_dir)
        writer(result, base)

    return {
        "text": txt,
        "json": jsn,
        "srt": os.path.join(run_dir, f"{base}.srt"),
        "vtt": os.path.join(run_dir, f"{base}.vtt"),
        "dir": run_dir,
    }