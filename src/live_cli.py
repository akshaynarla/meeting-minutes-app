# src/live_cli.py
from __future__ import annotations

import argparse
import queue
import sys

import numpy as np
import sounddevice as sd

from backend import LiveTranscriber


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live microphone transcription (offline-capable).")
    p.add_argument("--model", default="base", help="faster-whisper model size (e.g., tiny, base, small).")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps (GPU used if available).")
    p.add_argument("--compute-type", default="auto", help="auto|int8|float16|float32 (auto is recommended).")
    p.add_argument("--sample-rate", type=int, default=16000, help="Input sample rate for the mic stream.")
    p.add_argument("--block-ms", type=int, default=500, help="Audio block size in milliseconds.")
    p.add_argument("--vad-threshold", type=float, default=0.6, help="VAD threshold (tune for noisy rooms).")
    p.add_argument("--translate", action="store_true", help="Translate speech to English (any language → English).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    blocksize = int(args.sample_rate * args.block_ms / 1000)

    task = "translate" if args.translate else "transcribe"
    lt = LiveTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type=None if args.compute_type == "auto" else args.compute_type,
        vad_threshold=args.vad_threshold,
        sample_rate=16000,  # internal processing rate
        task=task,
    )

    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(status, file=sys.stderr)
        # indata is shape (frames, channels) float32
        q.put(indata.copy())

    mode = "TRANSLATE → English" if args.translate else "TRANSCRIBE"
    print(f"Mode: {mode}. Listening. Press Ctrl+C to stop.", flush=True)

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
                texts = lt.process_block(audio, args.sample_rate)
                for t in texts:
                    print(t, flush=True)

    except KeyboardInterrupt:
        # finalize any buffered speech
        for t in lt.flush():
            print(t, flush=True)
        print("\nStopped.", flush=True)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())