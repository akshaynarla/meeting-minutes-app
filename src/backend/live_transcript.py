# src/backend/live_transcript.py
from __future__ import annotations

import numpy as np
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False

from .device_utils import resolve_config, resolve_faster_whisper_device

class LiveTranscriber:
    """Minimal live ASR for microphone streaming.

    This implementation is intentionally offline-capable: it uses a Voice Activity Detector 
    VAD and transcribes finalized speech segments with faster-whisper (as the name indicates, a faster whisper).

    Notes:
      - Speaker diarization is NOT supported in live mode - was maybe a bit lazy?
      - Task for next iteration
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str | None = None,
        vad_threshold: float = 0.6,
        sample_rate: int = 16000,
        use_silero_vad: bool = True,
    ):
        cfg = resolve_config(device, compute_type)
        self.device = cfg.device
        fw_device = resolve_faster_whisper_device(self.device)

        # faster-whisper compute types differ slightly from WhisperX; keep safe defaults.
        fw_compute_type = cfg.compute_type
        if fw_device == "cpu" and fw_compute_type == "float16":
            fw_compute_type = "int8"

        self.asr = WhisperModel(model_size, device=fw_device, compute_type=fw_compute_type)

        self.sr = int(sample_rate)
        self.vad_threshold = float(vad_threshold)

        # VAD selection: Silero (default) or RMS (fallback)
        self.use_silero_vad = use_silero_vad and SILERO_VAD_AVAILABLE
        if self.use_silero_vad:
            self.vad_model = load_silero_vad()
        else:
            # RMS-based VAD (fallback if Silero VAD not available)
            # Map threshold (0.6) into a practical RMS threshold for float32 audio in [-1, 1].
            self.vad_rms_threshold = float(max(0.003, min(0.05, vad_threshold * 0.025)))

        self.min_speech_ms = 250
        self.min_silence_ms = 800
        self.max_segment_s = 20.0

        # Streaming state
        self._in_speech = False
        self._speech_buf = np.zeros((0,), dtype=np.float32)
        self._silence_samples = 0

    def _to_mono16k(self, x: np.ndarray, sr: int) -> np.ndarray:
        # x = float32 [-1, 1], shape (samples,) or (channels, samples) or (samples, channels)
        if x.ndim == 2:
            # common cases: (frames, channels) or (channels, frames)
            if x.shape[0] < x.shape[1]:
                # assume (channels, frames)
                x = x.mean(axis=0)
            else:
                # assume (frames, channels)
                x = x.mean(axis=1)
        x = np.asarray(x, dtype=np.float32)
        if sr != self.sr and x.size:
            x = resample_poly(x, self.sr, int(sr)).astype(np.float32, copy=False)
        return x

    def _is_speech_block(self, mono16k: np.ndarray) -> bool:
        if mono16k.size == 0:
            return False
        
        if self.use_silero_vad:
            # Silero VAD expects audio in range [-1, 1] at 16kHz
            # Returns timestamps of speech regions
            try:
                timestamps = get_speech_timestamps(
                    mono16k,
                    self.vad_model,
                    sampling_rate=self.sr,
                    threshold=self.vad_threshold,
                    num_steps_states=4,
                    use_onnx=False
                )
                return len(timestamps) > 0
            except Exception:
                # Fallback to RMS if Silero fails
                return self._is_speech_block_rms(mono16k)
        else:
            return self._is_speech_block_rms(mono16k)
    
    def _is_speech_block_rms(self, mono16k: np.ndarray) -> bool:
        """Fallback RMS-based voice detection."""
        rms = float(np.sqrt(np.mean(mono16k * mono16k)))
        return rms >= self.vad_rms_threshold

    @torch.inference_mode()
    def _transcribe_audio(self, mono16k: np.ndarray) -> str:
        # faster-whisper accepts numpy float32 in [-1, 1]
        segments, _ = self.asr.transcribe(
            mono16k.astype(np.float32, copy=False),
            beam_size=1,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip()).strip()

    @torch.inference_mode()
    def process_block(self, audio_f32: np.ndarray, sr: int) -> list[str]:
        """Push a small block of audio; returns a list of finalized transcribed text chunks.

        This returns text only when the VAD decides a speech segment has ended.
        """
        mono16k = self._to_mono16k(audio_f32, int(sr))
        if mono16k.size == 0:
            return []

        texts: list[str] = []

        is_speech = self._is_speech_block(mono16k)
        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_buf = np.zeros((0,), dtype=np.float32)
                self._silence_samples = 0

            self._speech_buf = np.concatenate([self._speech_buf, mono16k], axis=0)
            self._silence_samples = 0

            # force-cut very long continuous speech
            if self._speech_buf.size / self.sr >= self.max_segment_s:
                txt = self._transcribe_audio(self._speech_buf)
                if txt:
                    texts.append(txt)
                self._in_speech = False
                self._speech_buf = np.zeros((0,), dtype=np.float32)
                self._silence_samples = 0

            return texts

        # Silence
        if self._in_speech:
            self._speech_buf = np.concatenate([self._speech_buf, mono16k], axis=0)
            self._silence_samples += mono16k.size

            silence_ms = (self._silence_samples / self.sr) * 1000.0
            speech_ms = (self._speech_buf.size / self.sr) * 1000.0

            if silence_ms >= self.min_silence_ms and speech_ms >= self.min_speech_ms:
                # Trim trailing silence for transcription
                trim = int(min(self._silence_samples, self._speech_buf.size))
                speech_audio = self._speech_buf[:-trim] if trim else self._speech_buf

                if speech_audio.size:
                    txt = self._transcribe_audio(speech_audio)
                    if txt:
                        texts.append(txt)

                self._in_speech = False
                self._speech_buf = np.zeros((0,), dtype=np.float32)
                self._silence_samples = 0

        return texts

    def flush(self) -> list[str]:
        """Force-finalize any buffered speech (e.g., on Ctrl+C)."""
        if not self._in_speech or self._speech_buf.size == 0:
            return []
        speech_audio = self._speech_buf
        self._in_speech = False
        self._speech_buf = np.zeros((0,), dtype=np.float32)
        self._silence_samples = 0
        txt = self._transcribe_audio(speech_audio)
        return [txt] if txt else []

    def reset(self):
        """Call on session end to clear VAD internal state."""
        self._in_speech = False
        self._speech_buf = np.zeros((0,), dtype=np.float32)
        self._silence_samples = 0
