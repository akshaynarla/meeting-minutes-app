# src/live_translate_cli.py
"""CLI for live speech-to-text translation with optional TTS voice output.

Usage examples:
    # Text-only translation (works on CPU)
    python live_translate_cli.py --source en --target de --model qwen2.5:1.5b

    # With TTS voice output (best with GPU)
    python live_translate_cli.py --source en --target de --model qwen2.5:1.5b --tts

    # Custom TTS voice and Whisper model
    python live_translate_cli.py --source en --target fr --tts --speaker Carter --whisper-model small
"""
from __future__ import annotations

import argparse
import queue
import sys

import numpy as np
import sounddevice as sd

from backend import LiveTranslator, TTSEngine, SUPPORTED_LANGUAGES


def _lang_name(code: str) -> str:
    """Resolve a language code to its display name."""
    return SUPPORTED_LANGUAGES.get(code.lower(), code.title())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live microphone translation with optional TTS voice output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported languages: "
            + ", ".join(f"{k} ({v})" for k, v in sorted(SUPPORTED_LANGUAGES.items()))
        ),
    )

    # Translation
    p.add_argument("--source", default="en", help="Source language code (default: en).")
    p.add_argument("--target", default="de", help="Target language code (default: de).")

    # Ollama
    p.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model for translation.")
    p.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama endpoint.")

    # Whisper / STT
    p.add_argument("--whisper-model", default="base", help="faster-whisper model size.")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps.")
    p.add_argument("--compute-type", default="auto", help="auto|int8|float16|float32.")
    p.add_argument("--sample-rate", type=int, default=16000, help="Mic sample rate.")
    p.add_argument("--block-ms", type=int, default=500, help="Audio block size in ms.")
    p.add_argument("--vad-threshold", type=float, default=0.6, help="VAD sensitivity.")

    # TTS (optional)
    p.add_argument("--tts", action="store_true", help="Enable TTS voice output for translations.")
    p.add_argument("--voice", default=None, help="Piper TTS voice model (auto-selected by target language if omitted).")

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    blocksize = int(args.sample_rate * args.block_ms / 1000)

    source_name = _lang_name(args.source)
    target_name = _lang_name(args.target)

    # Initialize translator (STT + Ollama translation)
    translator = LiveTranslator(
        source_lang=source_name,
        target_lang=target_name,
        ollama_model=args.model,
        ollama_url=args.ollama_url,
        model_size=args.whisper_model,
        device=args.device,
        compute_type=None if args.compute_type == "auto" else args.compute_type,
        vad_threshold=args.vad_threshold,
        sample_rate=16000,
    )

    # Initialize TTS (optional)
    tts = None
    if args.tts:
        voice = args.voice or TTSEngine.suggest_voice(target_name)
        tts = TTSEngine(voice=voice)
        if tts.available:
            print(f"TTS enabled (voice: {voice}).", flush=True)
        else:
            print(
                "WARNING: Piper not installed — TTS disabled. "
                "Run: pip install piper-tts",
                file=sys.stderr,
            )
            tts = None

    # Audio input queue
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print(f"Translating: {source_name} → {target_name}", flush=True)
    print(f"Ollama model: {args.model}", flush=True)
    print("Listening. Press Ctrl+C to stop.\n", flush=True)

    try:
        with sd.InputStream(
            samplerate=args.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=callback,
        ):
            while True:
                block = q.get()
                audio = block[:, 0] if block.ndim == 2 else block
                chunks = translator.process_block(audio, args.sample_rate)
                for chunk in chunks:
                    print(f"[{source_name}] {chunk.original}", flush=True)
                    print(f"[{target_name}] {chunk.translated}\n", flush=True)

                    # Speak the translation if TTS is enabled
                    if tts is not None:
                        tts.play(chunk.translated, blocking=True)

    except KeyboardInterrupt:
        # Finalize any buffered speech
        for chunk in translator.flush():
            print(f"[{source_name}] {chunk.original}", flush=True)
            print(f"[{target_name}] {chunk.translated}\n", flush=True)
            if tts is not None:
                tts.play(chunk.translated, blocking=True)
        print("\nStopped.", flush=True)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
