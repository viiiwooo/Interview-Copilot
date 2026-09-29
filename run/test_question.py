"""Test the pipeline with a real interview question.

Records 15s from the C920 mic, runs VAD → ASR → LLM, and prints:
  * the transcribed question
  * the LLM's answer
  * latency breakdown

Usage:
  python -m run.test_question

>>> Speak a real interview question (e.g. "Как оптимизировать задержку API?") <<<
"""
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

from app.config import PipelineConfig
from app.main import Pipeline

MIC_DEVICE = 15  # C920
SAMPLE_RATE = 16000
DURATION = 15.0  # seconds


def main():
    print(f"Recording {DURATION}s from C920 mic (device {MIC_DEVICE})...")
    print(">>> Speak a real interview question now (e.g. 'Как оптимизировать задержку API?') <<<")
    time.sleep(1.5)
    frames = int(SAMPLE_RATE * DURATION)
    x = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="int16",
               device=MIC_DEVICE)
    sd.wait()
    x = x[:, 0]
    path = "run/test_question.wav"
    sf.write(path, x, SAMPLE_RATE, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
    peak = int(np.max(np.abs(x)))
    print(f"Saved {path} ({DURATION}s, rms={rms:.1f}, peak={peak})\n")

    print("Running pipeline (VAD → ASR → LLM)...")
    cfg = PipelineConfig()
    pipe = Pipeline(cfg)
    pipe.run_file(path, show_overlay=False)


if __name__ == "__main__":
    main()
