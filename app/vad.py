"""Silero VAD segmentation.

Two modes:
  * streaming  -- feed 512-sample frames as they arrive; emit a speech segment
                  (start/end in seconds) the moment a pause ends a phrase.
  * offline    -- run over a whole numpy array (file / benchmark) and return
                  all speech segments with timestamps.

Silero v6.2.3 expects 512-sample windows @16 kHz and forward(x, sr).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import silero_vad


@dataclass
class Segment:
    start: float          # seconds
    end: float            # seconds
    audio: np.ndarray     # int16 mono
    t_vad_done: float     # wall-clock when the segment was finalized (for latency)


class StreamingVAD:
    """Stateful VAD over a live stream of 512-sample int16 frames."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5,
                 min_speech_ms: int = 250, min_silence_ms: int = 300,
                 max_silence_ms: int = 1200, max_speech_s: float = 22.0,
                 partial_ms: int = 0):
        self.sr = sample_rate
        self.window = 512
        self.threshold = threshold
        self.min_speech = int(min_speech_ms / 1000 * sample_rate)
        self.min_silence = int(min_silence_ms / 1000 * sample_rate)
        self.max_silence = int(max_silence_ms / 1000 * sample_rate)
        self.max_speech = int(max_speech_s * sample_rate)
        # emit the in-progress speech every partial_ms for live transcription (0 = off)
        self.partial_step = int(partial_ms / 1000 * sample_rate)
        self._last_partial = 0
        self.model = silero_vad.load_silero_vad()
        self.model.eval()
        # rolling buffer of the current candidate speech region
        self._buf: list[np.ndarray] = []
        self._t_start = 0.0
        self._in_speech = False
        self._silence_run = 0
        self._pos = 0  # sample index of next frame

    def _probs(self, x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32) / 32768.0).unsqueeze(0)
        with torch.no_grad():
            p = self.model(t, self.sr)
        # p has shape (1, 1) or (1,); extract the scalar probability
        return p.item()

    def push(self, frame: np.ndarray, t_start: float, on_segment: Callable[[Segment], None],
             on_partial: Callable[[np.ndarray], None] | None = None) -> None:
        """Feed one 512-sample int16 frame; may emit a finalized segment.

        on_partial(audio) gets the speech collected so far, every partial_ms,
        while the phrase is still going (for a live, not yet final, transcript).
        """
        p = self._probs(frame)
        speech = p >= self.threshold
        frame_start = t_start

        if speech:
            if not self._in_speech:
                self._buf = [frame]
                self._t_start = frame_start
                self._in_speech = True
                self._silence_run = 0
                self._last_partial = 0
            else:
                self._buf.append(frame)
                self._silence_run = 0
            if on_partial is not None and self.partial_step:
                n = len(self._buf) * self.window
                if n - self._last_partial >= self.partial_step:
                    self._last_partial = n
                    on_partial(np.concatenate(self._buf))
        else:
            if self._in_speech:
                self._silence_run += self.window
                # keep trailing context so the tail isn't clipped
                self._buf.append(frame)
                speech_len = len(self._buf) * self.window
                # end condition: enough silence, or hard max length
                if self._silence_run >= self.min_silence or speech_len >= self.max_speech:
                    self._emit(frame_start, on_segment)

    def _emit(self, frame_start: float, on_segment: Callable[[Segment], None]) -> None:
        if not self._buf:
            return
        audio = np.concatenate(self._buf)
        # trim leading/trailing silence a touch (already ~30ms pad via threshold hysteresis)
        dur = len(audio) / self.sr
        if dur * self.sr < self.min_speech:
            self._reset()
            return
        seg = Segment(
            start=self._t_start,
            end=self._t_start + dur,
            audio=audio,
            t_vad_done=time.time(),
        )
        self._reset()
        on_segment(seg)

    def _reset(self) -> None:
        self._buf = []
        self._in_speech = False
        self._silence_run = 0
        self._last_partial = 0

    def flush(self, on_segment: Callable[[Segment], None]) -> None:
        """Force-emit any pending speech at stream end."""
        if self._in_speech and self._buf:
            self._emit(0.0, on_segment)
            self._reset()


def offline_segments(audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.5,
                     min_speech_ms: int = 250, min_silence_ms: int = 300) -> list[Segment]:
    """Run Silero VAD over a full int16 array; return speech segments."""
    model = silero_vad.load_silero_vad()
    model.eval()
    t = torch.from_numpy(audio.astype(np.float32) / 32768.0)
    with torch.no_grad():
        ts = silero_vad.get_speech_timestamps(
            t, model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )
    segs: list[Segment] = []
    now = time.time()
    for s in ts:
        a, b = int(s["start"]), int(s["end"])
        segs.append(Segment(start=a / sample_rate, end=b / sample_rate,
                            audio=audio[a:b], t_vad_done=now))
    return segs
