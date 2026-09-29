"""Probe a mic device: capture N seconds, report peak/RMS level.

Usage: python run/probe_mic_level.py [device] [seconds]
"""
from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd

sr = 16000


def main() -> None:
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    name = sd.query_devices(device).get("name", "?")
    print(f"device {device}: {name!r} — capturing {seconds:.0f}s ...", flush=True)
    with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                        device=device, blocksize=512) as stream:
        frames = []
        for _ in range(int(seconds * sr / 512)):
            data, _ = stream.read(512)
            frames.append(data[:, 0])
    audio = np.concatenate(frames)
    peak = int(np.max(np.abs(audio))) if len(audio) else 0
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if len(audio) else 0.0
    print(f"peak: {peak} ({peak / 32767:.1%} of full scale)")
    print(f"rms : {rms:.0f} ({rms / 32767:.1%} of full scale)")
    # with default gain x4 (MIC_GAIN), what would VAD see?
    amp = np.clip(audio.astype(np.float32) * 4.0, -32768, 32767)
    print(f"peak x4 : {int(np.max(np.abs(amp)))} ({np.max(np.abs(amp)) / 32767:.1%})")


if __name__ == "__main__":
    main()
