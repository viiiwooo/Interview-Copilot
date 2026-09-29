"""System-wide hotkey on Windows (user32.RegisterHotKey), stdlib ctypes only.

The copilot UI is usually NOT the focused window during an interview (the call
tab or app is), so a page keydown handler cannot see the key. RegisterHotKey
delivers WM_HOTKEY to a dedicated thread no matter which window has focus.
Note: while registered, the key is consumed and not seen by other apps.
"""
from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from typing import Callable

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000

_MODS = {"ALT": MOD_ALT, "CTRL": MOD_CONTROL, "CONTROL": MOD_CONTROL,
         "SHIFT": MOD_SHIFT, "WIN": MOD_WIN}
_KEYS = {f"F{i}": 0x6F + i for i in range(1, 13)}          # F1=0x70 .. F12=0x7B
_KEYS.update({"SPACE": 0x20, "ENTER": 0x0D, "PAUSE": 0x13, "SCROLLLOCK": 0x91,
              "INSERT": 0x2D, "HOME": 0x24, "END": 0x23,
              "`": 0xC0, "~": 0xC0, "Ё": 0xC0})  # tilde key, VK_OEM_3


def parse(spec: str) -> tuple[int, int]:
    """'Ctrl+Space' -> (MOD_CONTROL, VK_SPACE). Raises ValueError if unknown."""
    mods, vk = 0, None
    for part in spec.replace(" ", "").upper().split("+"):
        if part in _MODS:
            mods |= _MODS[part]
        elif part in _KEYS:
            vk = _KEYS[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part)
        else:
            raise ValueError(f"unknown key: {part}")
    if vk is None:
        raise ValueError(f"no key in hotkey: {spec}")
    return mods, vk


class GlobalHotkey:
    """Calls `callback` (on the hotkey thread) each time `spec` is pressed."""

    def __init__(self, spec: str, callback: Callable[[], None]):
        self.spec = spec
        self.callback = callback
        self.error = ""
        self._thread_id = 0
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Register the hotkey. False (and self.error set) if it failed."""
        if sys.platform != "win32":
            self.error = "глобальные клавиши поддерживаются только в Windows"
            return False
        try:
            mods, vk = parse(self.spec)
        except ValueError as e:
            self.error = str(e)
            return False
        self._thread = threading.Thread(target=self._run, args=(mods, vk), daemon=True)
        self._thread.start()
        self._ready.wait(2.0)
        return not self.error

    def _run(self, mods: int, vk: int) -> None:
        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, mods | MOD_NOREPEAT, vk):
            self.error = f"{self.spec} уже занята другой программой"
            self._ready.set()
            return
        self._ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    try:
                        self.callback()
                    except Exception as e:  # never kill the hotkey thread
                        print(f"[hotkey] callback failed: {e}", flush=True)
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self) -> None:
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._thread_id = 0
