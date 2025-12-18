# src/backend/backend_whisper.py
import os
import gc
import json
from datetime import datetime
from typing import Optional, Dict


import whisperx
# Some builds don’t expose DiarizationPipeline at top-level:
try:
    from whisperx.diarize import DiarizationPipeline  # preferred
except Exception:
    DiarizationPipeline = None  # handle later

def _format_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def process_audio(
    audio_path: str,
    out_dir: str = "outputs",
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    hf_token: Optional[str] = None,
    diarize: bool = False,
    language: Optional[str] = None
) -> Dict[str, str]:
    """
    Unified pipeline: Transcribe -> Align -> (Optional) Diarize -> Save
    Returns dict with paths: {"directory","text","json","conversation"}.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    base_name = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(out_dir, f"{base_name}_session")
    os.makedirs(run_dir, exist_ok=True)

    # 1) Transcribe
    asr_model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
    result = asr_model.transcribe(audio_path, batch_size=4)
    del asr_model; gc.collect()

    # 2) Alignment (accurate timestamps even without diarization)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio_path, device, return_char_alignments=False)
    del model_a; gc.collect()

    # 3) Diarization (optional)
    if diarize:
        if not hf_token:
            raise ValueError("Hugging Face Token (HF_TOKEN) is required for diarization.")
        if DiarizationPipeline is None:
            # Fall back if import failed for this WhisperX build
            try:
                from whisperx.diarize import DiarizationPipeline as _DP
                diarizer = _DP(use_auth_token=hf_token, device=device)
            except Exception as e:
                raise RuntimeError(f"Your whisperx build does not expose DiarizationPipeline: {e}")
        else:
            diarizer = DiarizationPipeline(use_auth_token=hf_token, device=device)

        diar_segments = diarizer(audio_path)
        result = whisperx.assign_word_speakers(diar_segments, result)
        del diarizer; gc.collect()

    # 4) Save outputs
    json_path = os.path.join(run_dir, "raw_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(run_dir, "transcript.txt")
    full_text = " ".join(seg["text"].strip() for seg in result["segments"])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    md_path = os.path.join(run_dir, "conversation.md")
    lines, current_speaker = [], None
    for seg in result["segments"]:
        start = _format_time(seg["start"])
        text = seg["text"].strip()
        speaker = seg.get("speaker", "Speaker") if diarize else "Speaker"
        if diarize and speaker == current_speaker and lines:
            lines[-1] += f" {text}"
        else:
            lines.append(f"**{speaker}** [{start}]: {text}")
            current_speaker = speaker
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"directory": run_dir, "text": txt_path, "json": json_path, "conversation": md_path}
