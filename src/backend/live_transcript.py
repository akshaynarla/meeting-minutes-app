# live_backend.py
from __future__ import annotations
import numpy as np
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

# Minimal live ASR:
#      - Silero VAD (VADIterator) to detect speech regions on streaming blocks
#      - faster-whisper to transcribe each closed speech chunk
class LiveTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",                  # or "cuda"
        compute_type: str | None = None,     
        vad_threshold: float = 0.6,
        sample_rate: int = 16000,
    ):
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        # load whisper model once
        self.asr = WhisperModel(model_size, device=device, compute_type=compute_type)

        # load silero vad + utils (one line, no custom VAD code)
        self.vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        (self.get_speech_timestamps, self.save_audio,
         self.read_audio, VADIterator, self.collect_chunks) = utils

        self.sr = sample_rate
        self.vad_iter = VADIterator(self.vad_model, sampling_rate=self.sr, threshold=vad_threshold)

    def _to_mono16k(self, x: np.ndarray, sr: int) -> np.ndarray:
        # x = float32 [-1, 1], shape (samples,) or (channels, samples)
        if x.ndim == 2:
            x = x.mean(axis=0)
        if sr != self.sr:
            x = resample_poly(x, self.sr, sr)
        return x.astype(np.float32, copy=False)

    @torch.inference_mode()
    def process_block(self, audio_f32: np.ndarray, sr: int) -> list[str]:
        """
        Push a small block of audio; returns a list of finalized transcribed text chunks.
        Only returns when Silero closes a speech segment in this block.
        """
        mono16k = self._to_mono16k(audio_f32, sr)
        if mono16k.size == 0:
            return []

        # VADIterator handles state; returns segments for THIS block (sample idx)
        t = torch.from_numpy(mono16k).float()
        speech_ts = self.vad_iter(t, return_seconds=False)

        # Convert those ts into actual audio chunks for THIS block
        chunks = self.collect_chunks(t, speech_ts)  # list of torch tensors (16k mono)
        texts: list[str] = []

        for ch in chunks:
            # faster-whisper accepts numpy float32 in [-1, 1]
            z = ch.numpy().astype(np.float32)
            segments, _ = self.asr.transcribe(
                z,
                beam_size=1, best_of=1,
                vad_filter=False,                 # we already did VAD
                condition_on_previous_text=False, # independent chunks
            )
            piece = " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip())
            if piece:
                texts.append(piece)

        return texts

    def reset(self):
        """Call on session end to clear VAD internal state."""
        self.vad_iter.reset_states()