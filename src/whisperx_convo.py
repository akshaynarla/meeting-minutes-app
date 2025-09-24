# one-shot ASR+alignment+diarization → conversation.md
import os, json, gc
import whisperx
from datetime import datetime

def transcribe_to_conversation(
    audio_path: str,
    out_dir: str = "outputs",
    model_size: str = "base",
    device: str = "cpu",                # "cpu" or "cuda"
    compute_type: str | None = None,    # None -> auto: "int8" on CPU, "float16" on CUDA
    hf_token: str | None = None,
    timestamps: bool = True,
) -> dict:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"

    base = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(out_dir, f"{base}_whisperx"); os.makedirs(run_dir, exist_ok=True)

    # 1) ASR
    asr = whisperx.load_model(model_size, device, compute_type=compute_type)
    result = asr.transcribe(audio_path)
    del asr; gc.collect()
    try:
        import torch; 
        if device == "cuda": torch.cuda.empty_cache()
    except Exception:
        pass

    # 2) Alignment (word-level times)
    align_model, meta = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio_path, device, return_char_alignments=False)
    del align_model; gc.collect()
    try:
        if device == "cuda": torch.cuda.empty_cache()
    except Exception:
        pass

    # 3) Diarization + assign speakers
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN (Hugging Face read token) for diarization.")
    diar = whisperx.diarize.DiarizationPipeline(use_auth_token=token, device=device)
    diar_segments = diar(audio_path)
    result = whisperx.assign_word_speakers(diar_segments, result)

    # Save raw outputs
    txt_path  = os.path.join(run_dir, f"{base}.txt")
    json_path = os.path.join(run_dir, f"{base}.json")
    with open(txt_path, "w", encoding="utf-8") as f:  f.write(result.get("text", "").strip() + "\n")
    with open(json_path, "w", encoding="utf-8") as f: json.dump(result, f, ensure_ascii=False, indent=2)

    # Build conversation.md by merging consecutive lines with the same speaker
    def hms(t):
        t = int(round(t));  return f"{t//3600:02}:{(t%3600)//60:02}:{t%60:02}"
    lines, last_spk, buf = [], None, ""
    for seg in result["segments"]:
        spk  = seg.get("speaker", "SPEAKER_00")
        text = (seg.get("text") or "").strip()
        if spk != last_spk:
            if buf: lines.append(buf.strip())
            ts  = f"[{hms(seg['start'])}] " if timestamps else ""
            buf = f"{ts}{spk}: {text}"
            last_spk = spk
        else:
            buf += " " + text
    if buf: lines.append(buf.strip())

    conv_path = os.path.join(run_dir, "conversation.md")
    with open(conv_path, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")

    return {"text": txt_path, "json": json_path, "conversation": conv_path, "dir": run_dir}