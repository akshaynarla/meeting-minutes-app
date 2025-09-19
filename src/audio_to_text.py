import argparse
import os
import json
import whisper

def transcribe(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Whisper model: {args.model} (CPU)…")
    model = whisper.load_model(args.model, device="cpu")

    print(f"Transcribing: {args.audio_path}")
    result = model.transcribe(
        args.audio_path,
        fp16=False,                 # set CPU mode
        language=args.language,     # None -> auto-detect
        task="translate" if args.translate else "transcribe",
        verbose=False
    )

    # Base filename without extension
    base = os.path.splitext(os.path.basename(args.audio_path))[0]
    out_txt = os.path.join(args.output_dir, f"{base}.txt")
    out_json = os.path.join(args.output_dir, f"{base}.json")

    # Plain text transcript
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(result["text"].strip() + "\n")

    # Full JSON (includes segments with start/end timestamps)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Subtitles (SRT + VTT) using Whisper writers: suitable for videos
    # for ext in ["srt", "vtt", "tsv"]:
    #    writer = get_writer(ext, args.output_dir)
    #    writer(result, base)

    # timestamped transcript
    if args.diarize_hint:
        ts_path = os.path.join(args.output_dir, f"{base}_timestamped.txt")
        with open(ts_path, "w", encoding="utf-8") as f:
            for seg in result.get("segments", []):
                start = seg["start"]
                end = seg["end"]
                text = seg["text"].strip()
                f.write(f"[{start:7.2f}–{end:7.2f}] {text}\n")

    print("Transcription Complete")
    print(f"• Text: {out_txt}")
    print(f"• JSON: {out_json}")
