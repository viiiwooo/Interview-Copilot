"""Loopback audio capture via sounddevice (PortAudio).

On Windows the browser's output is captured through the "Stereo Mix" /
"Loopback" device, or a virtual cable (VB-Cable / VoiceMeeter). We pick the
first device whose name matches a hint; fall back to the default output device
in loopback mode.

The capture thread pushes 512-sample (32 ms @16 kHz) frames into a bounded
queue so the VAD stage can process them in real time without blocking audio.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass
class AudioFrame:
    samples: np.ndarray  # int16 mono
    t_start: float       # seconds since capture start
    t_end: float


def find_loopback_device(hint: str | None = "Loopback") -> int | None:
    """Return device index whose name contains `hint` (case-insensitive)."""
    if hint is None:
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    hl = hint.lower()
    for i, d in enumerate(devices):
        name = d.get("name", "")
        if hl in name.lower():
            return i
    return None


class LoopbackCapture:
    """Continuously capture mono 16 kHz int16 audio into a queue."""

    def __init__(self, sample_rate: int = 16000, hint: str | None = "Loopback",
                 device_index: int | None = None, block_ms: int = 32):
        self.sample_rate = sample_rate
        self.block_frames = int(sample_rate * block_ms / 1000)  # 512
        self.q: queue.Queue[AudioFrame | None] = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self._device = device_index if device_index is not None else find_loopback_device(hint)
        self._t0 = 0.0
        self._stream = None
        self.device_index = self._device
        self.device_name = ""

    def start(self) -> None:
        if self._device is not None:
            self.device_name = sd.query_devices(self._device).get("name", "")
        else:
            # fall back to default output device, captured in loopback
            self.device_name = "(default output, loopback)"
        self._t0 = time.time()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_frames,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, t_info, status):
        if status:
            pass  # xruns are tolerable for VAD; just keep going
        data = indata[:, 0].copy()
        t = time.time() - self._t0
        # pad/truncate to exact block size for stable VAD windowing
        if len(data) < self.block_frames:
            data = np.pad(data, (0, self.block_frames - len(data)))
        else:
            data = data[: self.block_frames]
        frame = AudioFrame(samples=data, t_start=t, t_end=t + len(data) / self.sample_rate)
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            # drop oldest to stay real-time
            try:
                self.q.get_nowait()
                self.q.put_nowait(frame)
            except queue.Empty:
                pass

    def read(self, timeout: float = 0.2) -> AudioFrame | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None


def list_devices() -> str:
    """Human-readable device list for choosing the loopback source."""
    lines = []
    for i, d in enumerate(sd.query_devices()):
        lines.append(f"[{i}] {d.get('name','')}  in={d.get('max_input_channels')} out={d.get('max_output_channels')}")
    return "\n".join(lines)
