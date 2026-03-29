# src/backend/live_translate.py
"""Live Speech-to-Text Translation pipeline.

Chain:  Mic → VAD → faster-whisper STT → Ollama translation → text (+ optional TTS)

All processing happens locally — no audio or text leaves the device
(unless an explicit remote Ollama endpoint is configured).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .live_transcript import LiveTranscriber
from .backend_llm import _ollama_chat


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

_TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the following {source_lang} text to {target_lang}. "
    "Return ONLY the translation — no explanations, no annotations, "
    "no quotation marks. Preserve the original tone and meaning."
)

# Common language code → display name mapping (used in UI/CLI)
SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "sv": "Swedish",
}


def translate_text(
    text: str,
    source_lang: str = "English",
    target_lang: str = "German",
    model: str = "qwen2.5:1.5b",
    base_url: str = "http://localhost:11434",
    timeout: int = 30,
) -> str:
    """Translate a single piece of text using Ollama.

    Parameters
    ----------
    text : str
        Source text to translate.
    source_lang, target_lang : str
        Human-readable language names (e.g. "English", "German").
    model : str
        Ollama model name.
    base_url : str
        Ollama endpoint URL.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Translated text, or the original text on failure.
    """
    if not text or not text.strip():
        return ""

    system_prompt = _TRANSLATE_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text.strip()},
    ]

    try:
        result = _ollama_chat(
            base_url,
            model,
            messages,
            temperature=0.1,
            timeout=timeout,
            stream=False,
        )
        return result.strip() if result else text
    except Exception:
        # Graceful fallback: return original text so the pipeline
        # doesn't break if translation fails for one segment.
        return text


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranslatedChunk:
    """One finalized speech segment with its translation."""
    original: str
    translated: str


# ---------------------------------------------------------------------------
# LiveTranslator — orchestrator
# ---------------------------------------------------------------------------

class LiveTranslator:
    """Wraps LiveTranscriber + Ollama translation into a single call.

    Usage is identical to LiveTranscriber: push audio blocks via
    ``process_block()`` and receive translated text chunks back.
    """

    def __init__(
        self,
        source_lang: str = "English",
        target_lang: str = "German",
        ollama_model: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434",
        translate_timeout: int = 30,
        # LiveTranscriber params (forwarded)
        model_size: str = "base",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        vad_threshold: float = 0.6,
        sample_rate: int = 16000,
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.translate_timeout = translate_timeout

        # The internal STT engine — reuses your existing, proven pipeline.
        self._transcriber = LiveTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            vad_threshold=vad_threshold,
            sample_rate=sample_rate,
            task="transcribe",  # always transcribe; translation is done by Ollama
        )

    # ------------------------------------------------------------------
    # Public API — mirrors LiveTranscriber
    # ------------------------------------------------------------------

    def process_block(
        self, audio_f32: np.ndarray, sr: int
    ) -> List[TranslatedChunk]:
        """Push a small block of audio; returns translated chunks when
        the VAD detects end-of-speech."""
        texts = self._transcriber.process_block(audio_f32, sr)
        return self._translate_batch(texts)

    def flush(self) -> List[TranslatedChunk]:
        """Force-finalize any buffered speech and translate it."""
        texts = self._transcriber.flush()
        return self._translate_batch(texts)

    def reset(self):
        """Reset internal state."""
        self._transcriber.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _translate_batch(self, texts: List[str]) -> List[TranslatedChunk]:
        chunks: List[TranslatedChunk] = []
        for original in texts:
            if not original.strip():
                continue
            translated = translate_text(
                original,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                model=self.ollama_model,
                base_url=self.ollama_url,
                timeout=self.translate_timeout,
            )
            chunks.append(TranslatedChunk(original=original, translated=translated))
        return chunks
