from __future__ import annotations
from datetime import datetime
import os, json
from typing import Optional, Dict
import whisper
from whisper.utils import get_writer

def transcribe_whisper(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    translate: bool = False,
    output_dir: str = "outputs"
) -> Dict[str, str]:
    """
    Transcription with Whisper.(Slow without GPUs)
    Returns dict with paths: text, json, srt
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    base = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(output_dir, f"{base}_transcript"); os.makedirs(output_dir, exist_ok=True)

    model = whisper.load_model(model_size)
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

    # subtitle files
    for ext in ["srt"]:
        writer = get_writer(ext, run_dir)
        writer(result, base)

    return {
        "text": txt,
        "json": jsn,
        "srt": os.path.join(run_dir, f"{base}.srt"),
    }