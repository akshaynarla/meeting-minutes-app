import argparse
import audio_to_text
import text_to_meetmins

def main():
    parser = argparse.ArgumentParser(description="CPU-only Whisper transcription")
    parser.add_argument("audio_path", help="Path to your audio/video file (mp3, wav, m4a, mp4, etc.)")
    parser.add_argument("--model", default="base", choices=["tiny","base","small","medium","large"],
                        help="Whisper model size (CPU-friendly: tiny/base/small/medium). Default: base")
    parser.add_argument("--output_dir", default="transcripts", help="Output directory")
    parser.add_argument("--language", default=None, help="Force language code (e.g., 'en'). If omitted, auto-detect.")
    parser.add_argument("--translate", action="store_true",
                        help="Translate speech to English. Default:transcribing)")
    parser.add_argument("--diarize_hint", action="store_true",
                        help="Adds timestamps per segment; useful later for manual speaker labeling")
    parser.add_argument("transcript_txt", help="Path to transcript .txt (from Whisper)") # needs to be automatically done from audio_to_text
    parser.add_argument("--segments_json", default=None, help="Optional Whisper JSON to attach timestamps")
    parser.add_argument("--out_md", default="meeting_minutes.md")
    parser.add_argument("--out_csv", default="action_items.csv")
    parser.add_argument("--abstractive_model", default="sshleifer/distilbart-cnn-12-6",
                    help="HF model for abstractive summary (smaller than BART-large).")
    parser.add_argument("--summary_mode", choices=["auto","abstractive","extractive"], default="auto",
                    help="auto tries abstractive then falls back to LexRank.")
    parser.add_argument("--key_points", type=int, default=8, help="How many key sentences to list")
    args = parser.parse_args()

    audio_to_text.transcribe(args)
    text_to_meetmins.meeting_minutes(args)


if __name__ == "__main__":
    main()