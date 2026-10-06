"""GigaAM ASR stage (audio -> Russian text).

Wraps the `gigaam` package. Loads once, then transcribes short (<=~25s) int16
mono arrays. Times each transcription for the latency report.
"""
from __future__ import annotations

import io
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf


ASR_TARGET_PEAK = 0.9
ASR_MAX_GAIN = 20.0


def _normalize(x: np.ndarray) -> np.ndarray:
    """Peak-normalize one speech segment: quiet sources (a soft-spoken
    interviewer in a tab, a distant mic) are recognized much worse by GigaAM.
    Gain is capped so near-silence is not blown up into noise."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 1e-4:
        return x
    gain = min(ASR_TARGET_PEAK / peak, ASR_MAX_GAIN)
    return np.clip(x * gain, -1.0, 1.0) if gain > 1.0 else x


def resolve_device(device: str) -> str:
    """Normalize "gpu" to "cuda" and fall back to CPU when CUDA is missing."""
    device = (device or "cpu").strip().lower()
    if device == "gpu":
        device = "cuda"
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print(f"[asr] {device} unavailable (torch without CUDA?), using cpu", flush=True)
            return "cpu"
    return device


@dataclass
class ASRResult:
    text: str
    t_load: float      # one-time model load (seconds)
    t_infer: float     # this transcription (seconds)
    n_samples: int


class GigaAMASR:
    def __init__(self, model_name: str = "v3_e2e_rnnt", device: str = "cuda",
                 fp16_encoder: bool = True):
        import gigaam
        self._gigaam = gigaam
        device = resolve_device(device)
        # fp16 only helps (and only works) on the GPU
        fp16_encoder = fp16_encoder and device.startswith("cuda")
        t0 = time.time()
        self.model = gigaam.load_model(model_name, fp16_encoder=fp16_encoder, device=device)
        self.model_name = model_name
        self.device = device
        self.t_load = 0.0
        if device.startswith("cuda"):
            # the first CUDA call takes ~12s (kernel setup); pay it at load,
            # not on the first real phrase
            self.transcribe(np.zeros(16000, dtype=np.int16))
        self.t_load = time.time() - t0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe an int16 mono array. Returns text + timing."""
        # GigaAM.load_audio expects a file path (runs ffmpeg). Write a temp wav.
        audio_f = _normalize(audio.astype(np.float32) / 32768.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_f, sample_rate, subtype="PCM_16")
            path = f.name
        t0 = time.time()
        try:
            res = self.model.transcribe(path)
            text = res.text
        finally:
            import os
            try:
                os.remove(path)
            except OSError:
                pass
        t_infer = time.time() - t0
        return ASRResult(text=text, t_load=self.t_load, t_infer=t_infer,
                         n_samples=len(audio))
