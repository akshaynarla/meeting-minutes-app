from __future__ import annotations
import os, argparse
from audio_to_text import transcribe_whisper
from conversation_maker import (
    run_diarization, assign_speakers_to_whisper_segments,
    create_or_load_speaker_map, apply_speaker_map, write_conversation_markdown
)
from text_to_meetmins import make_minutes_from_text

''' parse_args(): used for running from CLI'''
def parse_args():
    ap = argparse.ArgumentParser(description="Meeting pipeline (transcribe → identify speakers → minutes)")
    ap.add_argument("audio", help="Path to input audio/video file (mp3/wav/m4a/mp4)")
    ap.add_argument("--out", default="outputs", help="Output root directory")

    # Transcription using whisper API. Downloads first time.
    ap.add_argument("--whisper_model", default="base", choices=["tiny","base","small","medium","large"])
    ap.add_argument("--language", default=None, help="Language code (e.g., en). Auto if omitted.")
    ap.add_argument("--translate", action="store_true", help="Translate to English instead of transcribe")

    # Diarization i.e. speaker identification using pyannote. Downloads first time.
    ap.add_argument("--do_diar", action="store_true", help="Enable speaker diarization")
    ap.add_argument("--num_speakers", type=int, default=0, help="Known #speakers (optional)")
    ap.add_argument("--hf_token", default=None, help="Hugging Face read token (or set HF_TOKEN)")

    # Minutes
    ap.add_argument("--summary_mode", choices=["auto","abstractive","extractive"], default="auto")
    ap.add_argument("--abstractive_model", default="sshleifer/distilbart-cnn-12-6")
    ap.add_argument("--key_points_n", type=int, default=8)

    # Stages
    ap.add_argument("--transcribe_only", action="store_true")
    ap.add_argument("--conversation_only", action="store_true")
    ap.add_argument("--minutes_only", action="store_true",
                    help="Use an existing transcript (.txt) instead of audio. Provide via --minutes_txt")

    ap.add_argument("--minutes_txt", default=None, help="If --minutes_only, path to transcript .txt")
    return ap.parse_args()

''' 
    main(): runs the entire pipeline
'''
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Minutes-only path; if arguments are set accordingly
    if args.minutes_only:
        if not args.minutes_txt:
            raise SystemExit("--minutes_only requires --minutes_txt pointing to a transcript.txt")
        m = make_minutes_from_text(
            transcript_text_path=args.minutes_txt,
            whisper_json_path=None,
            out_dir=args.out,
            summary_mode=args.summary_mode,
            abstractive_model=args.abstractive_model,
            key_points_n=args.key_points_n,
        )
        print("Minutes:", m["minutes_md"])
        print("Actions CSV:", m["actions_csv"])
        return

    # Transcribe/Translate
    t = transcribe_whisper(
        audio_path=args.audio,
        model_size=args.whisper_model,
        language=args.language,
        translate=args.translate,
        output_dir=args.out,
    )
    print("Transcript TXT:", t["text"])
    print("Transcript JSON:", t["json"])
    print("SRT:", t["srt"])
    # return/break if only transcription was needed
    if args.transcribe_only:
        return

    # 2) Style conversation by identifying speakers
    source_txt_for_minutes = t["text"]
    if args.do_diar:
        turns = run_diarization(
            audio_path=args.audio,
            hf_token=args.hf_token,
            num_speakers=(args.num_speakers or None),
        )
        merged = assign_speakers_to_whisper_segments(t["json"], turns)

        # Speaker map template under transcript dir
        speaker_map_path = os.path.join(t["dir"], "speaker_map.json")
        mapping = create_or_load_speaker_map(speaker_map_path, merged)
        merged_named = apply_speaker_map(merged, mapping)

        conv_path = os.path.join(t["dir"], "conversation.md")
        write_conversation_markdown(merged_named, conv_path, timestamps=True)
        print("Conversation MD:", conv_path)
        source_txt_for_minutes = conv_path

    if args.conversation_only:
        return

    # 3) Minutes
    m = make_minutes_from_text(
        transcript_text_path=source_txt_for_minutes,
        out_dir=args.out,
        model="llama3.1:8b",          # or mistral:7b-instruct, phi3:3.8b
        base_url="http://localhost:11434",
        title="Meeting Minutes",
    )
    print("Minutes:", m["minutes_md"])
    print("Minutes (TXT):", m["minutes_txt"])

if __name__ == "__main__":
    main()