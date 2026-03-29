# src/backend/tts_engine.py
"""TTS engine using Piper (fast, offline, CPU-friendly).

Piper is a lightweight neural text-to-speech system optimized for edge
devices.  It runs faster than real-time even on a Raspberry Pi.

Installation:
    pip install piper-tts

Voice models (~50-100 MB each) are downloaded automatically on first use
from Hugging Face: https://huggingface.co/rhasspy/piper-voices
"""
from __future__ import annotations

import io
import logging
import os
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import Piper — fully optional
# ---------------------------------------------------------------------------
try:
    from piper.voice import PiperVoice  # type: ignore
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

try:
    import sounddevice as sd  # already a project dependency
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


# Where to cache downloaded voice models
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "piper"

# Default voice model to use (English, medium quality, ~60 MB)
DEFAULT_VOICE = "en_US-amy-medium"

# Pre-configured voices for common languages (model name → sample rate)
# Users can pick a voice matching their translation target language.
# Full list: https://rhasspy.github.io/piper-samples/
PIPER_VOICES = {
    "en_US-amy-medium": {"lang": "English", "sr": 22050},
    "en_US-lessac-medium": {"lang": "English", "sr": 22050},
    "de_DE-thorsten-medium": {"lang": "German", "sr": 22050},
    "fr_FR-siwis-medium": {"lang": "French", "sr": 22050},
    "es_ES-davefx-medium": {"lang": "Spanish", "sr": 22050},
    "it_IT-riccardo-x_low": {"lang": "Italian", "sr": 16000},
    "pt_BR-faber-medium": {"lang": "Portuguese", "sr": 22050},
    "nl_NL-mls-medium": {"lang": "Dutch", "sr": 22050},
    "pl_PL-darkman-medium": {"lang": "Polish", "sr": 22050},
    "ru_RU-irinia-medium": {"lang": "Russian", "sr": 22050},
    "zh_CN-huayan-medium": {"lang": "Chinese", "sr": 22050},
    "tr_TR-dfki-medium": {"lang": "Turkish", "sr": 22050},
    "sv_SE-nst-medium": {"lang": "Swedish", "sr": 22050},
    "ko_KO-kagamine_rin-medium": {"lang": "Korean", "sr": 22050},
}

# Map target language name → suggested voice
LANG_TO_VOICE = {}
for _voice, _meta in PIPER_VOICES.items():
    _lang = _meta["lang"]
    if _lang not in LANG_TO_VOICE:
        LANG_TO_VOICE[_lang] = _voice


def _download_voice(voice_name: str, models_dir: Path) -> Path:
    """Download a Piper voice model from Hugging Face if not cached."""
    model_file = models_dir / f"{voice_name}.onnx"
    config_file = models_dir / f"{voice_name}.onnx.json"

    if model_file.exists() and config_file.exists():
        return model_file

    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine the HF path structure: lang/name/quality
    # e.g. en_US-amy-medium → en/en_US/amy/medium/en_US-amy-medium.onnx
    parts = voice_name.split("-")
    locale = parts[0]  # e.g. en_US
    lang_code = locale.split("_")[0]  # e.g. en
    speaker = parts[1] if len(parts) > 1 else "default"
    quality = parts[2] if len(parts) > 2 else "medium"

    base_url = (
        f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang_code}/{locale}/{speaker}/{quality}"
    )

    import requests

    for filename, target in [
        (f"{voice_name}.onnx", model_file),
        (f"{voice_name}.onnx.json", config_file),
    ]:
        url = f"{base_url}/{filename}"
        logger.info("Downloading %s ...", url)
        print(f"Downloading Piper voice: {filename} ...", flush=True)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info("Voice model cached at %s", model_file)
    return model_file


class TTSEngine:
    """Text-to-Speech engine using Piper (CPU-friendly, fully offline).

    The voice model is lazy-loaded on first ``synthesize()`` call.
    Models (~50-100 MB) are auto-downloaded from Hugging Face on first use.

    Parameters
    ----------
    voice : str
        Piper voice model name (e.g. "en_US-amy-medium").
    models_dir : str or Path, optional
        Directory to cache downloaded voice models.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        models_dir: Optional[str] = None,
    ):
        self._voice_name = voice
        self._models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self._voice: Optional[PiperVoice] = None  # lazy-loaded
        self._sample_rate: int = 22050

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if Piper is installed."""
        return PIPER_AVAILABLE

    @property
    def can_play(self) -> bool:
        """True if we can both synthesize and play audio."""
        return self.available and SOUNDDEVICE_AVAILABLE

    @property
    def voice_name(self) -> str:
        return self._voice_name

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_voice(self):
        """Lazy-load the voice model on first use."""
        if self._voice is not None:
            return

        if not self.available:
            raise RuntimeError(
                "Piper is not installed. Run: pip install piper-tts"
            )

        model_path = _download_voice(self._voice_name, self._models_dir)
        config_path = model_path.with_suffix(".onnx.json")

        logger.info("Loading Piper voice: %s", self._voice_name)
        self._voice = PiperVoice.load(
            str(model_path),
            config_path=str(config_path),
        )

        # Read sample rate from the loaded voice config
        if hasattr(self._voice, "config") and hasattr(self._voice.config, "sample_rate"):
            self._sample_rate = self._voice.config.sample_rate
        else:
            meta = PIPER_VOICES.get(self._voice_name, {})
            self._sample_rate = meta.get("sr", 22050)

        logger.info("Piper voice loaded (sr=%d).", self._sample_rate)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize speech from text.

        Returns
        -------
        np.ndarray or None
            Int16 PCM audio, or None if TTS is unavailable.
        """
        if not self.available:
            logger.debug("TTS unavailable — skipping synthesis.")
            return None

        if not text or not text.strip():
            return None

        self._ensure_voice()

        try:
            # Piper writes WAV data to a file-like object
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                self._voice.synthesize(text.strip(), wav)

            # Read back the raw PCM data
            buf.seek(0)
            with wave.open(buf, "rb") as wav:
                self._sample_rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)

            return audio
        except Exception as e:
            logger.warning("TTS synthesis failed: %s", e)
            return None

    def play(
        self,
        text: str,
        blocking: bool = True,
    ) -> bool:
        """Synthesize and immediately play through the default speaker.

        Returns
        -------
        bool
            True if audio was played successfully.
        """
        if not self.can_play:
            logger.debug("TTS playback unavailable — skipping.")
            return False

        audio = self.synthesize(text)
        if audio is None or audio.size == 0:
            return False

        try:
            # Convert int16 to float32 for sounddevice
            audio_f32 = audio.astype(np.float32) / 32768.0
            sd.play(audio_f32, samplerate=self._sample_rate)
            if blocking:
                sd.wait()
            return True
        except Exception as e:
            logger.warning("TTS playback failed: %s", e)
            return False

    def stop(self):
        """Stop any currently playing audio."""
        if SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except Exception:
                pass

    @staticmethod
    def suggest_voice(target_lang: str) -> str:
        """Suggest a Piper voice for the given target language name.

        Falls back to the default English voice if no match is found.
        """
        return LANG_TO_VOICE.get(target_lang, DEFAULT_VOICE)
