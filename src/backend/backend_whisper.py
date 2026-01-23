from __future__ import annotations

import os
import gc
import json
from datetime import datetime
from typing import Optional, Dict

import whisperx

from .device_utils import resolve_config

# diarization with whisperx, if library available, else no speaker diarization 
try:
    from whisperx.diarize import DiarizationPipeline
except Exception:
    DiarizationPipeline = None

# helper function to provide readable timestamps
def _format_time(seconds: float) -> str:
    m, s = divmod(float(seconds), 60.0)
    h, m = divmod(m, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

# vibe-coded: based on https://www.reddit.com/r/LocalLLaMA/comments/1edryd2/how_fast_big_llms_can_work_on_consumer_cpu_and/
def _default_batch_size(device: str, model_size: str) -> int:
    if device == "cpu":
        if str(model_size).startswith("large"):
            return 1
        return 2
    return 4


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
    """Transcribe a recorded meeting and optionally diarize speakers (i.e., identify speakers).

    Pipeline:
      1) Transcribe (WhisperX ASR)
      2) Align (WhisperX alignment)
      3) (Optional) Diarize (WhisperX / pyannote via HF)
      4) Write outputs (json, txt, conversation.md)

      - The function never uploads audio anywhere; any network usage is only for
        (optional) model downloads if they are not already cached locally.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    cfg = resolve_config(device, compute_type)
    device = cfg.device
    compute_type = cfg.compute_type

    base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{base_name}_session")
    os.makedirs(run_dir, exist_ok=True)

    # 1) Transcribe using WhisperX. Language is auto-identified by the model.
    batch_size = _default_batch_size(device, model_size)
    try:
        asr_model = whisperx.load_model(
            model_size,
            device,
            compute_type=compute_type,
            language=language
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load WhisperX model '{model_size}' on device='{device}' "
            f"with compute_type='{compute_type}'. Error: {e}"
        )

    try:
        result = asr_model.transcribe(audio_path, batch_size=batch_size)
    finally:
        del asr_model
        gc.collect()

    # 2) Alignment (improves timestamps; also used for speaker assignment)
    try:
        # forces the model to align speech with audio at a word level
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get("language"),
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_path,
            device,
            return_char_alignments=False
        )
    finally:
        try:
            del model_a
        except Exception:
            pass
        gc.collect()

    # 3) Diarization (identify speakers also)
    # Default uses: pyannote/speaker-diarization-3.1. Should be sufficient.
    if diarize:
        if not hf_token:
            raise ValueError("Hugging Face token is required for diarization.")
        try:
            diarizer = DiarizationPipeline(use_auth_token=hf_token, device=device)

            diar_segments = diarizer(audio_path)
            result = whisperx.assign_word_speakers(diar_segments, result)
        finally:
            try:
                del diarizer
            except Exception:
                pass
            gc.collect()

    # 4) Save outputs from the WhisperX framework for further processing
    # Here, transcript and the speaker/diarized transcript is made available in the output directory.
    json_path = os.path.join(run_dir, "raw_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(run_dir, "transcript.txt")
    full_text = " ".join(seg.get("text", "").strip() for seg in result.get("segments", [])).strip()
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text + ("\n" if full_text else ""))

    md_path = os.path.join(run_dir, "conversation.md")
    lines, current_speaker = [], None
    # Get the output in correct format by using the segments provided by the diarization.
    # Avoids for ex:
    # Alice [00:00:05]: Hello Alice [00:00:08]: how are you
    # Alice [00:00:05]: Hello how are you
    for seg in result.get("segments", []):
        start = _format_time(seg.get("start", 0.0))
        text = seg.get("text", "").strip()
        if not text:
            continue

        speaker = seg.get("speaker", "Speaker") if diarize else "Speaker"
        if diarize and speaker == current_speaker and lines:
            lines[-1] += f" {text}"
        else:
            lines.append(f"**{speaker}** [{start}]: {text}")
            current_speaker = speaker

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return {"directory": run_dir, "text": txt_path, "json": json_path, "conversation": md_path}
