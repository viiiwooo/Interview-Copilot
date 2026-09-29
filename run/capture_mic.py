"""Capture audio from the C920 microphone (device 15) and run the pipeline.

Usage:
  python run/capture_mic.py [duration_seconds]

Records `duration` seconds from the C920 mic, saves to run/mic_capture.wav,
then runs the full pipeline (VAD → ASR → LLM) and prints the latency report.
"""
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

from app.config import PipelineConfig
from app.main import Pipeline

MIC_DEVICE = 15  # C920 (highest peak in testing)
SAMPLE_RATE = 16000


def capture(duration: float = 10.0) -> str:
    """Record `duration` seconds from the C920 mic; return the wav path."""
    print(f"Recording {duration}s from C920 mic (device {MIC_DEVICE})...")
    print(">>> Speak Russian now <<<")
    time.sleep(1.0)
    frames = int(SAMPLE_RATE * duration)
    x = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16",
               device=MIC_DEVICE)
    sd.wait()
    x = x[:, 0]
    path = "run/mic_capture.wav"
    sf.write(path, x, SAMPLE_RATE, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
    peak = int(np.max(np.abs(x)))
    print(f"Saved {path} ({duration}s, rms={rms:.1f}, peak={peak})")
    return path


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    wav = capture(duration)

    print("\nRunning pipeline (VAD → ASR → LLM)...")
    cfg = PipelineConfig()
    pipe = Pipeline(cfg)
    pipe.run_file(wav, show_overlay=False)


if __name__ == "__main__":
    main()
