# src/backend/device_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    compute_type: str


def resolve_device(requested: Optional[str]) -> str:
    """Resolve a device string.

    Supported inputs:
      - None / "auto" / "": prefer GPU, else CPU
      - "cpu": respected if available; falls back to CPU if not
    """
    if requested is None:
        requested = "auto"

    req = str(requested).strip().lower()
    if req in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if req == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return "cpu"


def resolve_compute_type(device: str, requested: Optional[str]) -> str:
    """Resolve compute_type for Whisper/WhisperX and faster-whisper.

    Rules:
      - None / "auto" / "": float16 on CUDA/MPS, else int8 on CPU
      - otherwise: use requested verbatim
    """
    if requested is None:
        requested = "auto"
    req = str(requested).strip().lower()
    if req in ("", "auto"):
        return "float16" if device in ("cuda", "mps") else "int8"
    return str(requested)


def resolve_faster_whisper_device(device: str) -> str:
    """faster-whisper supports 'cpu' and 'cuda'. Treat MPS as CPU."""
    return "cuda" if device == "cuda" else "cpu"


def resolve_config(device: Optional[str], compute_type: Optional[str]) -> DeviceConfig:
    d = resolve_device(device)
    ct = resolve_compute_type(d, compute_type)
    return DeviceConfig(device=d, compute_type=ct)
