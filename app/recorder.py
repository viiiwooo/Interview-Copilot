"""Meeting recording: transcript (Markdown) + optional audio per session.

One session = one folder sessions/YYYY-MM-DD_HH-MM-SS/ with
  transcript.md  -- every recognized phrase and every question/answer, appended live
  audio.wav      -- the captured 16 kHz mono stream (if audio saving is on)
  review.md      -- the LLM review of the dialog (written by the web server)
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import soundfile as sf


class SessionRecorder:
    def __init__(self, root: Path, sample_rate: int = 16000):
        self.root = root
        self.sample_rate = sample_rate
        self.enabled = False        # the UI checkbox
        self.save_audio = True
        self.dir: Path | None = None  # current (active) session folder
        self.last_dir: Path | None = None
        self._audio: sf.SoundFile | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.dir is not None

    # ---- lifecycle ----------------------------------------------------------
    def start(self, source: str) -> Path:
        """Open a new session folder (no-op if one is already open)."""
        with self._lock:
            if self.dir is not None:
                return self.dir
            name = time.strftime("%Y-%m-%d_%H-%M-%S")
            d = self.root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "transcript.md").write_text(
                f"# Встреча {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"- Источник звука: {source}\n\n## Стенограмма\n\n",
                encoding="utf-8")
            if self.save_audio:
                self._audio = sf.SoundFile(d / "audio.wav", "w", self.sample_rate, 1, "PCM_16")
            self.dir = d
            return d

    def stop(self) -> Path | None:
        with self._lock:
            d = self.dir
            if self._audio is not None:
                try:
                    self._audio.close()
                except Exception:
                    pass
                self._audio = None
            if d is not None:
                self._append_locked(f"\n---\n_Запись остановлена {time.strftime('%H:%M:%S')}_\n")
                self.last_dir = d
            self.dir = None
            return d

    # ---- content ------------------------------------------------------------
    def _append_locked(self, text: str) -> None:
        if self.dir is None:
            return
        with open(self.dir / "transcript.md", "a", encoding="utf-8") as f:
            f.write(text)

    def add_phrase(self, text: str, speaker: str) -> None:
        with self._lock:
            self._append_locked(f"**[{time.strftime('%H:%M:%S')}] {speaker}:** {text}\n\n")

    def add_qa(self, question: str, answer: str, backend: str) -> None:
        quoted = "\n".join("> " + line for line in (answer.strip() or "(пустой ответ)").splitlines())
        with self._lock:
            self._append_locked(
                f"**[{time.strftime('%H:%M:%S')}] Подсказка ({backend})** на вопрос «{question}»:\n\n"
                f"{quoted}\n\n")

    def add_audio(self, pcm16: bytes) -> None:
        with self._lock:
            if self._audio is not None:
                import numpy as np
                self._audio.write(np.frombuffer(pcm16, dtype="int16"))

    # ---- sessions -----------------------------------------------------------
    def sessions(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if (d / "transcript.md").exists():
                out.append({"name": d.name,
                            "active": d == self.dir,
                            "review": (d / "review.md").exists(),
                            "audio": (d / "audio.wav").exists()})
        return out

    def session_dir(self, name: str | None) -> Path | None:
        """Folder by name; default = the active session, else the latest one."""
        if name:
            d = (self.root / name).resolve()
            # only folders directly inside sessions/ (no path tricks)
            if d.parent == self.root.resolve() and (d / "transcript.md").exists():
                return d
            return None
        if self.dir is not None:
            return self.dir
        items = self.sessions()
        return self.root / items[0]["name"] if items else None
