"""Transparent-ish topmost overlay window (tkinter) that floats over the browser.

Two panes: the interviewer's question (transcribed) and the suggested answer
(streamed token-by-token). A small status line shows pipeline latency.

The window is topmost so it stays above the browser tab. It is NOT transparent
(tkinter transparency is unreliable on Windows), but it is small, draggable,
and sits in a corner so it doesn't cover the video call.
"""
from __future__ import annotations

import re
import threading
import tkinter as tk
from dataclasses import dataclass


def md_to_plain(md: str) -> str:
    """Markdown -> readable plain text for the Tk overlay (no markup symbols)."""
    out = []
    in_code = False
    for line in md.replace("\r", "").split("\n"):
        if re.match(r"^\s*```", line):
            in_code = not in_code
            continue
        if in_code:
            out.append("    " + line)
            continue
        if re.match(r"^\s*\|?[\s:|-]+\|?\s*$", line) and "-" in line and "|" in line:
            continue  # table separator row
        line = re.sub(r"^\s*#{1,6}\s+(.*?)\s*#*\s*$", lambda m: m.group(1).upper(), line)
        line = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", line)
        line = re.sub(r"^\s*>\s?", "  ", line)
        if line.strip().startswith("|"):
            line = "  ".join(c.strip() for c in line.strip().strip("|").split("|"))
        line = re.sub(r"\*\*|__", "", line)
        line = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]*?\S)\*(?![\w*])", r"\1", line)
        line = line.replace("`", "")
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


@dataclass
class OverlayState:
    question: str = ""
    answer: str = ""
    status: str = "ожидание речи..."
    latency: str = ""


class Overlay:
    def __init__(self, width: int = 760, height: int = 520, x: int = 90, y: int = 90):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.root = None
        self.q_var = None
        self.a_var = None
        self.s_var = None
        self.l_var = None
        self._built = False
        self._answer_md = ""   # raw Markdown of the current answer

    def build(self) -> None:
        if self._built:
            return
        self.root = tk.Tk()
        self.root.title("Interview Copilot")
        self.root.geometry(f"{self.width}x{self.height}+{self.x}+{self.y}")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0f1420")

        # make it draggable by the header
        header = tk.Label(self.root, text="  🎧 Interview Copilot   (перетащи за заголовок)",
                          bg="#1a2233", fg="#8ab4ff", anchor="w",
                          font=("Consolas", 11, "bold"))
        header.pack(fill="x")
        header.bind("<Button-1>", self._drag_start)
        header.bind("<B1-Motion>", self._drag_move)
        header.bind("<Button-3>", lambda e: self.root.destroy())

        # status line
        self.s_var = tk.StringVar(value="ожидание речи...")
        self.l_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.s_var, bg="#0f1420", fg="#5fd08a",
                 anchor="w", font=("Consolas", 10)).pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(self.root, textvariable=self.l_var, bg="#0f1420", fg="#5a6b8a",
                 anchor="w", font=("Consolas", 9)).pack(fill="x", padx=8)

        # question pane
        tk.Label(self.root, text="ВОПРОС", bg="#0f1420", fg="#ffd27a",
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        self.q_var = tk.StringVar(value="")
        q = tk.Text(self.root, height=5, bg="#141b2b", fg="#e8eefc",
                    wrap="word", font=("Consolas", 11), relief="flat",
                    insertbackground="#8ab4ff")
        q.pack(fill="both", expand=False, padx=10, pady=(2, 6))
        q.bind("<B1-Motion>", lambda e: "break")
        q.configure(state="disabled")
        self._q_text = q

        # answer pane
        tk.Label(self.root, text="ОТВЕТ (подсказка)", bg="#0f1420", fg="#8ab4ff",
                 font=("Consolas", 9, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        self.a_var = tk.StringVar(value="")
        a = tk.Text(self.root, height=14, bg="#101726", fg="#cfe3ff",
                    wrap="word", font=("Consolas", 11), relief="flat",
                    insertbackground="#8ab4ff")
        a.pack(fill="both", expand=True, padx=10, pady=(2, 8))
        a.configure(state="disabled")
        self._a_text = a

        self._built = True

    # ---- thread-safe updates (called from worker threads) -----------------
    def set_question(self, text: str) -> None:
        self._post(lambda: (self._q_text.configure(state="normal"),
                            self._q_text.delete("1.0", "end"),
                            self._q_text.insert("1.0", text),
                            self._q_text.configure(state="disabled")))

    def set_answer(self, text: str, append: bool = False) -> None:
        # keep the raw Markdown and redraw it as plain text: a markup token
        # (e.g. "**") may be split across streamed chunks
        self._answer_md = self._answer_md + text if append else text
        plain = md_to_plain(self._answer_md)

        def _do():
            self._a_text.configure(state="normal")
            self._a_text.delete("1.0", "end")
            self._a_text.insert("end", plain)
            self._a_text.see("end")
            self._a_text.configure(state="disabled")
        self._post(_do)

    def set_status(self, text: str) -> None:
        self._post(lambda: self.s_var.set(text))

    def set_latency(self, text: str) -> None:
        self._post(lambda: self.l_var.set(text))

    def _post(self, fn) -> None:
        if self.root is None:
            return
        try:
            self.root.after(0, fn)
        except Exception:
            pass

    # ---- drag ------------------------------------------------------------
    def _drag_start(self, e):
        self._dx = e.x
        self._dy = e.y

    def _drag_move(self, e):
        if self._dx is None:
            return
        x = self.root.winfo_x() + (e.x - self._dx)
        y = self.root.winfo_y() + (e.y - self._dy)
        self.root.geometry(f"+{x}+{y}")

    def run(self) -> None:
        """Start the tkinter mainloop (blocking). Call from the main thread."""
        self.build()
        self.root.mainloop()

    def destroy(self) -> None:
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
