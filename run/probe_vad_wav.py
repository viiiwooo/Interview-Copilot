"""Probe a WAV: peak level + Silero VAD segments at several thresholds.

Usage: python run/probe_vad_wav.py <wav> [thresholds ...]
"""
from __future__ import annotations

import sys

import numpy as np
import soundfile as sf
import torch
import silero_vad


def main() -> None:
    wav = sys.argv[1]
    thresholds = [float(x) for x in sys.argv[2:]] or [0.3, 0.5]
    data, sr = sf.read(wav, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import scipy.signal as sps
        data = sps.resample_poly(data, 16000, sr)
        sr = 16000
    audio = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    peak = int(np.max(np.abs(audio))) if len(audio) else 0
    print(f"{wav}: {len(audio)/sr:.1f}s @ {sr}Hz, peak {peak} ({peak/32767:.1%})")

    model = silero_vad.load_silero_vad()
    model.eval()
    # frame-level probabilities: 512-sample windows @16kHz
    n_frames = len(audio) // 512
    probs = []
    with torch.no_grad():
        for i in range(n_frames):
            chunk = audio[i * 512:(i + 1) * 512].astype(np.float32) / 32768.0
            p = model(torch.from_numpy(chunk).unsqueeze(0), sr)
            probs.append(p.item())
    p = torch.tensor(probs)
    print(f"VAD prob: mean {p.mean():.3f}  max {p.max():.3f}  "
          f"frames>0.3: {(p > 0.3).sum()}/{len(p)}  frames>0.5: {(p > 0.5).sum()}/{len(p)}")
    t = torch.from_numpy(audio.astype(np.float32) / 32768.0)
    for thr in thresholds:
        with torch.no_grad():
            ts = silero_vad.get_speech_timestamps(
                t, model, threshold=thr, sampling_rate=sr,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300)
        print(f"  threshold {thr}: {len(ts)} segments "
              f"{[(int(s['start'])/sr, int(s['end'])/sr) for s in ts][:8]}")


if __name__ == "__main__":
    main()
