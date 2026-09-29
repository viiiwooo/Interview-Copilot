"""Web interface for the interview copilot.

FastAPI + WebSocket:
  * `/` — HTML page from app/static/index.html (Markdown-rendered answers,
    live mic controls, browser mic check; all client-side)
  * `/ws` — WebSocket endpoint (real-time updates)
  * `/api/run` — POST: run the pipeline over a WAV file, stream results via WS
  * `/api/live` — POST: start live mic capture (selectable device and gain) in real time
  * `/api/live/stop` — POST: stop live capture
  * `/api/devices` — GET: list input audio devices

Usage:
  python -m app.web  # starts on http://localhost:9090
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import (PipelineConfig, LOGS, RUN, WEB_UI_PORT, ACCUMULATE_DEFAULT, SEND_HOTKEY,
                     SESSIONS, RECORD_DEFAULT, RECORD_AUDIO, REVIEW_PROMPT,
                     REVIEW_MAX_TOKENS, REVIEW_MAX_CHARS)
from .main import Pipeline, _load_wav_int16
from .vad import offline_segments, StreamingVAD
from .asr import GigaAMASR
from .llm_client import ask, resolve_model
from .deepseek_web import DeepSeekWebClient, is_available as deepseek_is_available
from .deepseek_ext import ExtBridge, bridge_available
from .hotkey import GlobalHotkey
from .recorder import SessionRecorder

app = FastAPI(title="Interview Copilot")

# LLM backend selection: "local" (llama.cpp), "deepseek" (CDP) or "deepseek_ext"
# (Chrome extension on chat.deepseek.com — no debug port needed).
LLM_BACKENDS = ("local", "deepseek", "deepseek_ext")

# Shared state (thread-safe via lock)
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "ожидание..."
        self.question = ""
        self.answer = ""
        self.latency = ""
        self.clients: set[WebSocket] = set()
        self.pipeline: Pipeline | None = None
        self.cfg = PipelineConfig()
        # live mic capture state
        self.live_active = False
        self.live_device: int | None = None
        self.live_gain: float | None = None
        self.live_stop: threading.Event | None = None
        self.asr: GigaAMASR | None = None
        self.llm_model: str = ""
        self.vad: StreamingVAD | None = None
        # LLM backend selection (UI dropdown): "local" or "deepseek"
        self.llm_backend: str = "local"
        self.cdp_port: int = 9222
        self._ds_client: DeepSeekWebClient | None = None
        # Chrome-extension backend bridge (deepseek_ext)
        self.ext_bridge = ExtBridge()
        # browser live capture: audio arrives as binary /ws messages (Web Audio
        # in the browser is cleaner than the server-side PortAudio capture)
        self.bactive = False
        self.bframe_q: queue.Queue | None = None   # raw int16 frames from the browser
        self.bseg_q: queue.Queue | None = None     # finalized VAD segments
        self.bvad: StreamingVAD | None = None
        self.bstop: threading.Event | None = None
        self.bws: WebSocket | None = None          # the socket that started the capture
        self.bdebug: list[bytes] | None = None    # raw frames, saved to wav on stop
        # event loop of the uvicorn thread (WebSocket.send_text is async and
        # must be scheduled onto it when called from worker threads)
        self.loop: asyncio.AbstractEventLoop | None = None
        # accumulate mode: recognized speech is collected in `pending` and sent
        # to the LLM as one question on a (global) hotkey or UI button
        self.accumulate: bool = ACCUMULATE_DEFAULT
        self.pending: str = ""
        self.hotkey_spec: str = SEND_HOTKEY
        self.hotkey: GlobalHotkey | None = None
        self.hotkey_error: str = ""
        # one answer at a time: live segments and manual sends share the LLM
        self.answer_lock = threading.Lock()
        # final and live (partial) transcription share one GigaAM model
        self.asr_lock = threading.Lock()
        # meeting recording (transcript.md + audio.wav per session) and review
        self.recorder = SessionRecorder(SESSIONS)
        self.recorder.enabled = RECORD_DEFAULT
        self.recorder.save_audio = RECORD_AUDIO
        self.rec_speaker = "Интервьюер"     # who is heard by the current capture
        self.bsource = "mic"                # browser capture source: tab | mic | both
        self.review_prompt = REVIEW_PROMPT
        self.review_running = False

state = AppState()


@app.on_event("startup")
async def _capture_loop() -> None:
    state.loop = asyncio.get_running_loop()


class RunRequest(BaseModel):
    wav: str  # path to a WAV file


class LiveRequest(BaseModel):
    device: int | None = None  # None -> cfg.mic_device (C920)
    gain: float | None = None  # None -> cfg.mic_gain (MIC_GAIN)


def broadcast(msg: dict) -> None:
    """Send a message to all connected WebSocket clients.

    WebSocket.send_text is async; when broadcast is called from a worker
    thread (pipeline / live capture) the coroutine must be scheduled onto
    the uvicorn event loop via run_coroutine_threadsafe.
    """
    text = json.dumps(msg, ensure_ascii=False)
    loop = state.loop
    with state.lock:
        clients = list(state.clients)
    for ws in clients:
        try:
            if loop is not None:
                asyncio.run_coroutine_threadsafe(ws.send_text(text), loop)
            else:
                ws.send_text(text)
        except Exception:
            with state.lock:
                state.clients.discard(ws)


def set_status(text: str) -> None:
    state.status = text
    broadcast({"type": "status", "text": text})


def set_question(text: str) -> None:
    state.question = text
    broadcast({"type": "question", "text": text})


def set_answer(text: str, append: bool = False) -> None:
    if append:
        state.answer += text
    else:
        state.answer = text
    broadcast({"type": "answer", "text": state.answer})


def set_latency(text: str) -> None:
    state.latency = text
    broadcast({"type": "latency", "text": text})


class PartialASR:
    """Live transcript of the phrase that is still being spoken.

    The VAD hands over the speech collected so far (every PARTIAL_ASR_MS); only
    the newest snapshot is kept, older ones are dropped. A snapshot is skipped
    when the ASR is busy with a final segment, so answers are never delayed.
    Results for a phrase that already ended (seq changed) are discarded.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._job: tuple[int, np.ndarray] | None = None
        self._seq = 0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, audio: np.ndarray) -> None:
        with self._cv:
            self._job = (self._seq, audio)
            self._cv.notify()

    def end_phrase(self) -> None:
        """The phrase ended (final segment emitted): drop pending live results."""
        with self._cv:
            self._seq += 1
            self._job = None

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                seq, audio = self._job
                self._job = None
            asr = state.asr
            if asr is None or not state.asr_lock.acquire(blocking=False):
                continue  # the final ASR has priority
            try:
                text = asr.transcribe(audio, state.cfg.sample_rate).text.strip()
            except Exception as e:
                print(f"[partial] asr failed: {e}", flush=True)
                continue
            finally:
                state.asr_lock.release()
            with self._cv:
                current = seq == self._seq
            if current and text:
                broadcast({"type": "partial", "text": text})


partial_asr = PartialASR()


def process_segment(seg, cfg: PipelineConfig, asr: GigaAMASR, llm_model: str) -> None:
    """Process one speech segment: ASR → LLM → broadcast."""
    set_status(f"речь {seg.start:.1f}-{seg.end:.1f}s → распознаю...")
    t0 = time.time()
    with state.asr_lock:
        r = asr.transcribe(seg.audio, cfg.sample_rate)
    t_asr = time.time()
    text = r.text.strip()
    broadcast({"type": "partial", "text": ""})  # the final text replaces the live line
    peak = int(np.max(np.abs(seg.audio))) if len(seg.audio) else 0
    print(f"[asr] seg {seg.start:.1f}-{seg.end:.1f}s "
          f"({len(seg.audio)/cfg.sample_rate:.1f}s, peak {peak}) -> {text!r}",
          flush=True)

    if not text:
        # Noise/empty segment: keep the previous question and the in-progress
        # answer, just report that the segment was skipped. Save the clip so
        # the cause (startup click / noise floor / short word) can be checked.
        clip = RUN / f"empty_seg_{seg.start:.1f}.wav"
        sf.write(clip, seg.audio.astype(np.float32) / 32768.0,
                 cfg.sample_rate, subtype="PCM_16")
        print(f"[asr] seg {seg.start:.1f}-{seg.end:.1f}s empty — skipped, "
              f"clip saved: {clip}", flush=True)
        set_status(f"сегмент {seg.start:.1f}-{seg.end:.1f}s пустой (шум) — пропущен "
                   f"(клип: {clip.name})")
        return

    if state.recorder.active:
        state.recorder.add_phrase(text, state.rec_speaker)
    if state.accumulate:
        _add_pending(text)
        return
    answer_question(text, cfg, llm_model, t_asr=r.t_infer)


def answer_question(text: str, cfg: PipelineConfig, llm_model: str,
                    t_asr: float | None = None) -> None:
    """Send one question to the selected LLM and stream the answer to the UI.

    t_asr is None for a manual send (accumulated text), which has no ASR step.
    """
    with state.answer_lock:
        set_question(text)
        set_answer("")
        backend = state.llm_backend
        src = f"вопрос готов ({t_asr:.1f}s)" if t_asr is not None else "вопрос отправлен"
        if backend in ("deepseek", "deepseek_ext"):
            label = "DeepSeek (расширение)" if backend == "deepseek_ext" else "DeepSeek (web)"
            set_status(f"{src} → {label}...")
        else:
            set_status(f"{src} → отвечаю...")

        def on_token(tok: str):
            set_answer(tok, append=True)

        if backend == "deepseek_ext":
            llm = _ask_deepseek_ext(text, on_token)
        elif backend == "deepseek":
            llm = _ask_deepseek(text, on_token)
        else:
            if not llm_model:
                llm_model = state.llm_model = resolve_model(cfg.llm_base_url)
            llm = ask(cfg.llm_base_url, llm_model, cfg.llm_system_prompt, text,
                      cfg.llm_max_tokens, cfg.llm_temperature, cfg.llm_top_p, on_token)
        asr_part = f"ASR {t_asr:.1f}s · " if t_asr is not None else ""
        set_status(f"готово · {asr_part}LLM ttft {llm.ttft:.1f}s · total {llm.total:.1f}s")
        set_latency(f"{asr_part}LLM ttft {llm.ttft:.2f}s | всего {llm.total:.2f}s")
        if state.recorder.active:
            state.recorder.add_qa(text, llm.text, _backend_label(backend))


# ---- accumulate mode --------------------------------------------------------

def _broadcast_pending() -> None:
    broadcast({"type": "pending", "text": state.pending, "enabled": state.accumulate,
               "hotkey": state.hotkey_spec, "hotkey_error": state.hotkey_error})


def _add_pending(text: str) -> None:
    with state.lock:
        state.pending = (state.pending + " " + text).strip()
        n = len(state.pending)
    _broadcast_pending()
    set_status(f"накоплено {n} симв. — {state.hotkey_spec} или Пробел: отправить")


def send_pending(text: str | None = None) -> bool:
    """Send the accumulated text (or the UI-edited `text`) as one question."""
    with state.lock:
        q = (state.pending if text is None else text).strip()
        state.pending = ""
    _broadcast_pending()
    if not q:
        set_status("нечего отправлять: накопленный текст пуст")
        return False
    print(f"[pending] send: {q!r}", flush=True)
    threading.Thread(target=_answer_safe, args=(q,), daemon=True).start()
    return True


def _answer_safe(q: str) -> None:
    try:
        answer_question(q, state.cfg, state.llm_model)
    except Exception as e:
        set_status(f"ошибка отправки: {e}")


def _set_accumulate(enabled: bool, hotkey: str | None = None) -> None:
    """Toggle accumulate mode; the global hotkey is registered only while on."""
    if state.hotkey is not None:
        state.hotkey.stop()
        state.hotkey = None
    if hotkey:
        state.hotkey_spec = hotkey
    state.accumulate = enabled
    state.hotkey_error = ""
    if enabled:
        hk = GlobalHotkey(state.hotkey_spec, lambda: send_pending())
        if hk.start():
            state.hotkey = hk
        else:
            state.hotkey_error = hk.error
    _broadcast_pending()


def _fallback_local(question: str, on_token, reason: str) -> "LLMResult":
    """Answer with the local llama.cpp after a DeepSeek backend failed."""
    from .llm_client import LLMResult

    set_status(f"DeepSeek недоступен ({reason}) → локальный LLM")
    set_answer("")  # drop a partial DeepSeek answer before the local one streams in
    cfg = state.cfg
    if not state.llm_model:
        try:
            state.llm_model = resolve_model(cfg.llm_base_url)
        except Exception as e2:
            set_status(f"ни DeepSeek, ни локальный LLM недоступны: {e2}")
            return LLMResult(text="", ttft=0, total=0, n_tokens=0, tps=0)
    return ask(cfg.llm_base_url, state.llm_model, cfg.llm_system_prompt, question,
               cfg.llm_max_tokens, cfg.llm_temperature, cfg.llm_top_p, on_token)


def _ask_deepseek(question: str, on_token) -> "LLMResult":
    """Send a question to chat.deepseek.com via the user's Chrome (CDP)."""
    from .deepseek_web import DeepSeekWebClient
    if state._ds_client is None:
        state._ds_client = DeepSeekWebClient(state.cdp_port)
    try:
        prompt = state.cfg.deepseek_prompt_template.format(question=question)
        return state._ds_client.ask(prompt, on_token=on_token, on_replace=set_answer)
    except Exception as e:
        return _fallback_local(question, on_token, str(e))


def _ask_deepseek_ext(question: str, on_token) -> "LLMResult":
    """Send a question to chat.deepseek.com via the installed Chrome extension."""
    from .deepseek_ext import make_llm_result

    cfg = state.cfg
    prompt = cfg.deepseek_prompt_template.format(question=question)
    try:
        # bridge.ask is async; run it on the uvicorn loop so it can await the WS.
        data = asyncio.run_coroutine_threadsafe(
            state.ext_bridge.ask(prompt, on_token=on_token,
                                 timeout_s=cfg.deepseek_ext_timeout_s,
                                 on_replace=set_answer),
            state.loop,
        ).result(timeout=cfg.deepseek_ext_timeout_s + 15)
        return make_llm_result(data)
    except Exception as e:
        return _fallback_local(question, on_token, str(e))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    with state.lock:
        state.clients.add(ws)
    try:
        while True:
            msg = await ws.receive()
            mtype = msg.get("type", "")
            if mtype == "websocket.disconnect":
                break
            if msg.get("bytes"):
                # int16 PCM frames from the browser mic (512 samples @16 kHz)
                _handle_audio_frame(msg["bytes"])
            elif msg.get("text"):
                _handle_control(ws, msg["text"])
    except WebSocketDisconnect:
        pass
    finally:
        with state.lock:
            state.clients.discard(ws)
            was_capturing = state.bws is ws
            if was_capturing:
                state.bws = None
        # the capture source went away: stop the VAD/worker threads
        if was_capturing:
            _browser_capture_stop()


STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index():
    # read on every request: edits to index.html show up on browser reload
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    """Run the pipeline over a WAV file (blocking, in a thread)."""
    if not Path(req.wav).exists():
        return JSONResponse({"error": f"file not found: {req.wav}"}, status_code=404)

    def _run():
        cfg = state.cfg
        set_status(f"загружаю {Path(req.wav).name}...")
        audio, sr = _load_wav_int16(req.wav)
        set_status(f"{len(audio)/sr:.1f}s аудио @ {sr}Hz")

        # VAD
        t0 = time.time()
        segs = offline_segments(audio, sr, cfg.vad_threshold,
                                cfg.vad_min_speech_ms, cfg.vad_min_silence_ms)
        set_status(f"VAD: {len(segs)} сегментов за {time.time()-t0:.1f}s")

        if not segs:
            set_status("речь не найдена")
            return

        # ASR
        set_status("загружаю GigaAM...")
        t0 = time.time()
        asr = GigaAMASR(cfg.gigaam_model, cfg.gigaam_device, cfg.gigaam_fp16_encoder)
        set_status(f"GigaAM готов за {time.time()-t0:.1f}s")

        # LLM
        llm_model = resolve_model(cfg.llm_base_url)
        set_status(f"LLM: {Path(llm_model).name}")

        # Process each segment
        rec_dir = _rec_start("file")
        for i, seg in enumerate(segs):
            set_status(f"сегмент {i+1}/{len(segs)}: {seg.start:.1f}-{seg.end:.1f}s")
            process_segment(seg, cfg, asr, llm_model)
        _rec_stop(rec_dir)

        set_status("готово")

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "wav": req.wav}


class AccumulateRequest(BaseModel):
    enabled: bool
    hotkey: str | None = None


class PendingRequest(BaseModel):
    text: str | None = None


@app.get("/api/accumulate")
async def get_accumulate():
    return {"enabled": state.accumulate, "hotkey": state.hotkey_spec,
            "hotkey_error": state.hotkey_error, "pending": state.pending}


@app.post("/api/accumulate")
async def post_accumulate(req: AccumulateRequest):
    _set_accumulate(req.enabled, (req.hotkey or "").strip() or None)
    return await get_accumulate()


@app.post("/api/pending")
async def post_pending(req: PendingRequest):
    """Replace the accumulated text with the version edited in the UI."""
    with state.lock:
        state.pending = (req.text or "").strip()
    return {"status": "ok"}


@app.post("/api/pending/send")
async def post_pending_send(req: PendingRequest):
    return {"sent": send_pending(req.text)}


@app.post("/api/pending/clear")
async def post_pending_clear():
    with state.lock:
        state.pending = ""
    _broadcast_pending()
    return {"status": "ok"}


# ---- meeting recording ------------------------------------------------------

_SPEAKERS = {"tab": "Интервьюер", "mic": "Микрофон", "both": "Созвон",
             "server_mic": "Микрофон", "file": "Файл"}
_SOURCES = {"tab": "вкладка браузера (собеседник)", "mic": "микрофон (браузер)",
            "both": "вкладка + микрофон (обе стороны)",
            "server_mic": "микрофон (сервер)", "file": "WAV-файл"}


def _backend_label(backend: str) -> str:
    return {"deepseek_ext": "DeepSeek, расширение", "deepseek": "DeepSeek, CDP"}.get(
        backend, "локальный LLM")


def _broadcast_rec() -> None:
    r = state.recorder
    broadcast({"type": "rec", "enabled": r.enabled, "active": r.active,
               "audio": r.save_audio, "session": r.dir.name if r.dir else "",
               "last": r.last_dir.name if r.last_dir else ""})


def _rec_start(source: str) -> Path | None:
    """Open a session if recording is on. Returns its folder (for _rec_stop)."""
    if not state.recorder.enabled:
        return None
    state.rec_speaker = _SPEAKERS.get(source, "Собеседник")
    d = state.recorder.start(_SOURCES.get(source, source))
    print(f"[rec] session {d}", flush=True)
    _broadcast_rec()
    return d


def _rec_stop(rec_dir: Path | None) -> None:
    """Close the session opened by this capture (a newer one is left alone)."""
    if rec_dir is not None and state.recorder.dir == rec_dir:
        state.recorder.stop()
        print(f"[rec] saved {rec_dir}", flush=True)
        _broadcast_rec()


class RecordRequest(BaseModel):
    enabled: bool
    audio: bool | None = None


@app.get("/api/record")
async def get_record():
    r = state.recorder
    return {"enabled": r.enabled, "active": r.active, "audio": r.save_audio,
            "session": r.dir.name if r.dir else "",
            "last": r.last_dir.name if r.last_dir else ""}


@app.post("/api/record")
async def post_record(req: RecordRequest):
    r = state.recorder
    if req.audio is not None:
        r.save_audio = req.audio
    r.enabled = req.enabled
    if not r.enabled and r.active:
        r.stop()
    elif r.enabled and not r.active and (state.bactive or state.live_active):
        # capture already running: start recording right away
        _rec_start(state.bsource if state.bactive else "server_mic")
    _broadcast_rec()
    return await get_record()


@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": state.recorder.sessions(), "root": str(SESSIONS)}


@app.get("/api/sessions/{name}/{kind}")
async def get_session_file(name: str, kind: str):
    d = state.recorder.session_dir(name)
    if d is None or kind not in ("transcript", "review"):
        return JSONResponse({"error": "not found"}, status_code=404)
    f = d / f"{kind}.md"
    if not f.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"name": name, "path": str(f), "text": f.read_text(encoding="utf-8")}


# ---- dialog review (LLM -> review.md) -----------------------------------------

class ReviewRequest(BaseModel):
    session: str | None = None


class ReviewPromptRequest(BaseModel):
    prompt: str


@app.get("/api/review_prompt")
async def get_review_prompt():
    return {"prompt": state.review_prompt, "default": REVIEW_PROMPT}


@app.post("/api/review_prompt")
async def set_review_prompt(req: ReviewPromptRequest):
    state.review_prompt = req.prompt.strip() or REVIEW_PROMPT
    return {"status": "ok"}


def _review_llm(system: str, user: str, on_token) -> str:
    """Run the review on the selected backend (DeepSeek falls back to local)."""
    cfg = state.cfg
    backend = state.llm_backend
    if backend in ("deepseek_ext", "deepseek"):
        try:
            full = system + "\n\n" + user
            if backend == "deepseek_ext":
                data = asyncio.run_coroutine_threadsafe(
                    state.ext_bridge.ask(full, on_token=on_token, timeout_s=600,
                                         on_replace=lambda t: on_token(None, t)),
                    state.loop).result(timeout=620)
                return data["text"]
            if state._ds_client is None:
                state._ds_client = DeepSeekWebClient(state.cdp_port)
            return state._ds_client.ask(full, on_token=on_token, timeout_s=600,
                                        on_replace=lambda t: on_token(None, t)).text
        except Exception as e:
            on_token(None, "")  # drop a partial DeepSeek review
            set_status(f"DeepSeek недоступен ({e}) → разбор локальным LLM")
    if not state.llm_model:
        state.llm_model = resolve_model(cfg.llm_base_url)
    return ask(cfg.llm_base_url, state.llm_model, system, user, REVIEW_MAX_TOKENS,
               0.3, cfg.llm_top_p, on_token).text


def _run_review(d: Path) -> None:
    buf = {"text": ""}

    def on_token(tok, replace=None):
        buf["text"] = replace if replace is not None else buf["text"] + tok
        broadcast({"type": "review", "session": d.name, "text": buf["text"], "done": False})

    try:
        transcript = (d / "transcript.md").read_text(encoding="utf-8")
        note = ""
        if len(transcript) > REVIEW_MAX_CHARS:
            transcript = transcript[-REVIEW_MAX_CHARS:]
            note = "\n\n_Стенограмма длинная: в разбор попали последние " \
                   f"{REVIEW_MAX_CHARS} символов._"
        set_status(f"разбор встречи {d.name}...")
        with state.answer_lock:  # one LLM job at a time (DeepSeek composer, llama slot)
            text = _review_llm(state.review_prompt, "Стенограмма:\n\n" + transcript, on_token)
        text = (text or buf["text"]).strip()
        out = d / "review.md"
        out.write_text(f"# Разбор встречи {d.name}\n\n{text}{note}\n", encoding="utf-8")
        broadcast({"type": "review", "session": d.name, "text": text + note, "done": True,
                   "path": str(out)})
        set_status(f"разбор сохранён: {out}")
        print(f"[review] saved {out}", flush=True)
    except Exception as e:
        broadcast({"type": "review", "session": d.name, "text": buf["text"], "done": True,
                   "error": str(e)})
        set_status(f"ошибка разбора: {e}")
    finally:
        state.review_running = False


@app.post("/api/review")
async def post_review(req: ReviewRequest):
    d = state.recorder.session_dir(req.session)
    if d is None:
        return JSONResponse({"error": "нет записанных встреч"}, status_code=404)
    if state.review_running:
        return JSONResponse({"error": "разбор уже идёт"}, status_code=409)
    state.review_running = True
    threading.Thread(target=_run_review, args=(d,), daemon=True).start()
    return {"started": True, "session": d.name}


@app.get("/api/status")
async def get_status():
    return {
        "status": state.status,
        "question": state.question,
        "answer": state.answer,
        "latency": state.latency,
        "live_active": state.live_active,
        "live_device": state.live_device,
        "live_gain": state.live_gain,
        "browser_active": state.bactive,
        "accumulate": state.accumulate,
        "pending": state.pending,
    }


# ---- live mic capture -------------------------------------------------------

def _broadcast_live(running: bool, device: int | None, gain: float | None = None) -> None:
    broadcast({"type": "live", "running": running, "device": device, "gain": gain})


_SEG_SENTINEL = object()


def _live_loop(device: int, stop: threading.Event, gain: float) -> None:
    """Capture mono 16 kHz audio from `device`, feed VAD, process segments.

    The whole body is under try/finally: any failure (setup or capture) must
    clear the shared state, otherwise /api/live would return 409 forever.
    """
    seg_queue: queue.Queue = queue.Queue()
    worker_started = False
    try:
        import sounddevice as sd
        cfg = state.cfg

        set_status("загружаю GigaAM...")
        if state.asr is None:
            t0 = time.time()
            state.asr = GigaAMASR(cfg.gigaam_model, cfg.gigaam_device, cfg.gigaam_fp16_encoder)
            set_status(f"GigaAM готов за {time.time()-t0:.1f}s")
        if not state.llm_model:
            state.llm_model = resolve_model(cfg.llm_base_url)
            set_status(f"LLM: {Path(state.llm_model).name}")
        # local VAD instance: a previous capture winding down must not feed ours
        vad = StreamingVAD(cfg.sample_rate, cfg.vad_threshold, cfg.vad_min_speech_ms,
                           cfg.vad_min_silence_ms, cfg.vad_max_silence_ms, cfg.vad_max_speech_s,
                           partial_ms=cfg.partial_asr_ms)
        state.vad = vad

        def on_segment(seg):
            # queue segments: a single worker processes them in order, so a
            # late noise segment cannot clobber an in-progress answer
            partial_asr.end_phrase()
            seg_queue.put(seg)

        def worker_loop() -> None:
            while True:
                seg = seg_queue.get()
                if seg is _SEG_SENTINEL:
                    break
                try:
                    process_segment(seg, cfg, state.asr, state.llm_model)
                except Exception as e:
                    set_status(f"ошибка обработки сегмента: {e}")
            _rec_stop(rec_dir)

        rec_dir = _rec_start("server_mic")
        worker = threading.Thread(target=worker_loop, daemon=True)
        worker.start()
        worker_started = True

        set_status(f"слушаю микрофон (device {device}, gain ×{gain:g})...")
        print(f"[live] capturing from device {device} (gain {gain:g})", flush=True)
        clip_run = 0
        clip_warned = False
        with sd.InputStream(samplerate=cfg.sample_rate, channels=1, dtype="int16",
                            device=device, blocksize=512) as stream:
            while not stop.is_set():
                data, _ = stream.read(512)
                frame = data[:, 0]
                if gain != 1.0:
                    # amplify the quiet mic, clip to int16 range
                    frame = np.clip(frame.astype(np.float32) * gain,
                                    -32768, 32767).astype(np.int16)
                if not clip_warned and len(frame):
                    # ~0.5s of clipped peaks: the gain is too high for this mic
                    clip_run = clip_run + 1 if int(np.max(np.abs(frame))) >= 32700 else 0
                    if clip_run >= 16:
                        clip_warned = True
                        print(f"[live] clipping detected (gain {gain:g}) — lower the gain",
                              flush=True)
                        set_status(f"⚠ клиппирование: gain ×{gain:g} слишком большой — "
                                   f"уменьшите чувствительность")
                if state.recorder.active:
                    state.recorder.add_audio(frame.tobytes())
                vad.push(frame, time.time(), on_segment, partial_asr.submit)
            vad.flush(on_segment)
    except Exception as e:
        set_status(f"ошибка захвата: {e}")
    finally:
        if worker_started:
            # let the worker drain the queue, then stop it
            seg_queue.put(_SEG_SENTINEL)
        # only clear shared state if this thread is still the current one
        # (a newer capture may have replaced us while we were winding down)
        with state.lock:
            still_current = state.live_stop is stop
            if still_current:
                state.live_active = False
                state.live_gain = None
        _broadcast_live(False, device, None)
        if still_current:
            set_status("захват остановлен")


# ---- browser live capture (audio arrives over the existing /ws) -----------

def _broadcast_bcap(running: bool) -> None:
    broadcast({"type": "bcap", "running": running})


def _handle_audio_frame(data: bytes) -> None:
    """Queue one int16 PCM frame from the browser; dropped if not capturing."""
    q = state.bframe_q
    if q is None:
        return
    q.put(data)
    if state.bdebug is not None:
        state.bdebug.append(data)
    if state.recorder.active:
        state.recorder.add_audio(data)


def _handle_control(ws: WebSocket, text: str) -> None:
    """Browser control messages: capture_start / capture_stop."""
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return
    mtype = msg.get("type")
    if mtype == "capture_start":
        with state.lock:
            state.bws = ws
        _browser_capture_start(str(msg.get("source") or "mic"))
    elif mtype == "capture_stop":
        _browser_capture_stop()


def _browser_capture_start(source: str = "mic") -> None:
    """Start browser-side capture. Setup (ASR/LLM/VAD) runs off the event loop."""
    with state.lock:
        if state.bactive:
            return
        state.bactive = True
        state.bframe_q = queue.Queue()
        state.bseg_q = queue.Queue()
        state.bvad = None
        state.bstop = threading.Event()
        state.bdebug = []
        state.bsource = source
    threading.Thread(target=_browser_setup, args=(source,), daemon=True).start()


def _save_browser_debug(chunks: list[bytes] | None) -> Path | None:
    """Save the raw frames received from the browser (16 kHz mono int16) so
    the exact signal the server heard can be listened to: if the clip is
    clean, the problem is downstream (VAD/ASR); if distorted, in the browser."""
    if not chunks:
        return None
    pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
    path = RUN / f"browser_cap_{time.time():.0f}.wav"
    sf.write(path, pcm, state.cfg.sample_rate, subtype="PCM_16")
    print(f"[browser-capture] debug clip saved: {path} ({len(pcm) / 16000:.1f}s)",
          flush=True)
    return path


def _browser_setup(source: str = "mic") -> None:
    """Load ASR/LLM + VAD for the browser capture, start push/worker threads."""
    cfg = state.cfg
    frame_q = state.bframe_q
    seg_q = state.bseg_q
    stop = state.bstop
    try:
        set_status("загружаю GigaAM...")
        if state.asr is None:
            t0 = time.time()
            state.asr = GigaAMASR(cfg.gigaam_model, cfg.gigaam_device, cfg.gigaam_fp16_encoder)
            set_status(f"GigaAM готов за {time.time()-t0:.1f}s")
        if not state.llm_model:
            state.llm_model = resolve_model(cfg.llm_base_url)
            set_status(f"LLM: {Path(state.llm_model).name}")
        if not state.bactive:
            return  # stopped during the load: the queues are abandoned
        vad = StreamingVAD(cfg.sample_rate, cfg.vad_threshold, cfg.vad_min_speech_ms,
                           cfg.vad_min_silence_ms, cfg.vad_max_silence_ms, cfg.vad_max_speech_s,
                           partial_ms=cfg.partial_asr_ms)
        state.bvad = vad

        def on_segment(seg):
            partial_asr.end_phrase()
            seg_q.put(seg)

        def push_loop() -> None:
            # dedicated thread: the Silero forward must not block the event loop
            while True:
                try:
                    data = frame_q.get(timeout=0.5)
                except queue.Empty:
                    if stop.is_set():
                        break
                    continue
                vad.push(np.frombuffer(data, dtype=np.int16), time.time(), on_segment,
                         partial_asr.submit)
            vad.flush(on_segment)
            seg_q.put(_SEG_SENTINEL)

        rec_dir = _rec_start(source)

        def worker_loop() -> None:
            while True:
                seg = seg_q.get()
                if seg is _SEG_SENTINEL:
                    break
                try:
                    process_segment(seg, cfg, state.asr, state.llm_model)
                except Exception as e:
                    set_status(f"ошибка обработки сегмента: {e}")
            _rec_stop(rec_dir)  # after the last phrase of this capture is written

        threading.Thread(target=push_loop, daemon=True).start()
        threading.Thread(target=worker_loop, daemon=True).start()
        if source == "tab":
            set_status("слушаю вкладку собеседования...")
        elif source == "both":
            set_status("слушаю вкладку и микрофон...")
        else:
            set_status("слушаю микрофон (браузер)...")
        print(f"[browser-capture] started, source={source}", flush=True)
        _broadcast_bcap(True)
    except Exception as e:
        _browser_capture_stop()
        set_status(f"ошибка захвата: {e}")


def _browser_capture_stop() -> None:
    with state.lock:
        if not state.bactive:
            return
        state.bactive = False
        stop = state.bstop
        state.bvad = None
        state.bframe_q = None
        state.bseg_q = None
        state.bstop = None
        debug = state.bdebug
        state.bdebug = None
    if stop is not None:
        # the push thread exits, flushes pending speech, then signals the worker
        stop.set()
    clip = _save_browser_debug(debug)
    _broadcast_bcap(False)
    if clip is not None:
        set_status(f"захват остановлен (клип: {clip.name})")
    else:
        set_status("захват остановлен")


@app.get("/api/devices")
async def get_devices():
    """List input audio devices for the live-capture selector."""
    import sounddevice as sd
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            devices.append({"index": i, "name": d.get("name", ""),
                            "channels": d["max_input_channels"]})
    return {"devices": devices, "default": state.cfg.mic_device}


@app.post("/api/live")
async def start_live(req: LiveRequest):
    with state.lock:
        if state.live_active:
            return JSONResponse({"error": "захват уже запущен"}, status_code=409)
        # stop any previous capture still winding down, then replace its event
        if state.live_stop is not None:
            state.live_stop.set()
        state.live_stop = threading.Event()
        state.live_active = True
    device = req.device if req.device is not None else state.cfg.mic_device
    gain = req.gain if req.gain is not None else state.cfg.mic_gain
    gain = min(max(gain, 0.1), 10.0)
    state.live_device = device
    state.live_gain = gain
    _broadcast_live(True, device, gain)
    threading.Thread(target=_live_loop, args=(device, state.live_stop, gain), daemon=True).start()
    return {"status": "started", "device": device, "gain": gain}


class PromptRequest(BaseModel):
    prompt: str


@app.get("/api/prompt")
async def get_system_prompt():
    return {"prompt": state.cfg.llm_system_prompt}


@app.post("/api/prompt")
async def set_system_prompt(req: PromptRequest):
    """Set the LLM system prompt for this session (sent once from the browser)."""
    if req.prompt.strip():
        state.cfg.llm_system_prompt = req.prompt.strip()
    return {"status": "ok"}


class VadConfigRequest(BaseModel):
    min_silence_ms: int
    threshold: float | None = None


@app.post("/api/vad_config")
async def set_vad_config(req: VadConfigRequest):
    """Adjust VAD end-of-phrase silence (ms) and speech threshold for live capture."""
    state.cfg.vad_min_silence_ms = max(50, min(req.min_silence_ms, 5000))
    if req.threshold is not None:
        state.cfg.vad_threshold = max(0.05, min(req.threshold, 0.95))
    return {"status": "ok", "min_silence_ms": state.cfg.vad_min_silence_ms,
            "threshold": state.cfg.vad_threshold}


class LlmBackendRequest(BaseModel):
    backend: str          # "local" | "deepseek"
    cdp_port: int = 9222


@app.post("/api/llm_backend")
async def set_llm_backend(req: LlmBackendRequest):
    """Switch the answer LLM between local llama.cpp and a DeepSeek web backend."""
    backend = req.backend if req.backend in LLM_BACKENDS else "local"
    state.llm_backend = backend
    state.cdp_port = max(1, min(req.cdp_port or 9222, 65535))
    # reset the cached CDP client so a new port is picked up
    state._ds_client = None
    if backend == "deepseek":
        ok, info = deepseek_is_available(state.cdp_port)
        set_status(f"LLM: DeepSeek (web/CDP) — {'готово' if ok else 'недоступно: ' + info}")
    elif backend == "deepseek_ext":
        ok, info = bridge_available(state.ext_bridge)
        set_status(f"LLM: DeepSeek (расширение) — {info}")
    else:
        set_status("LLM: локальный llama.cpp")
    return {"status": "ok", "backend": backend, "cdp_port": state.cdp_port}


@app.get("/api/llm_backend")
async def get_llm_backend():
    """Current LLM backend + availability of each DeepSeek web backend."""
    if state.llm_backend == "deepseek":
        ok, info = deepseek_is_available(state.cdp_port)
    elif state.llm_backend == "deepseek_ext":
        ok, info = bridge_available(state.ext_bridge)
    else:
        ok, info = False, ""
    return {"backend": state.llm_backend, "cdp_port": state.cdp_port,
            "deepseek_ok": ok, "deepseek_info": info}


# ---- DeepSeek extension WebSocket ------------------------------------------
@app.websocket("/api/deepseek_ws")
async def deepseek_ext_ws(ws: WebSocket):
    """The content script on chat.deepseek.com connects here and stays attached."""
    await ws.accept()
    state.ext_bridge.attach(ws)
    broadcast({"type": "ext", "connected": True})
    try:
        while True:
            raw = await ws.receive_text()
            # single reader of the socket; ask() consumes from the bridge's queue
            state.ext_bridge.enqueue_incoming(raw)
    except WebSocketDisconnect:
        pass
    finally:
        if state.ext_bridge.detach(ws):
            broadcast({"type": "ext", "connected": False})


@app.get("/api/deepseek_ext")
async def get_deepseek_ext():
    """Whether the extension tab is connected (for the UI indicator)."""
    ok, info = bridge_available(state.ext_bridge)
    return {"connected": ok, "info": info}


@app.post("/api/live/stop")
async def stop_live():
    if state.live_stop is not None:
        state.live_stop.set()
        return {"status": "stopping"}
    return JSONResponse({"error": "захват не запущен"}, status_code=404)

def main():
    import uvicorn
    print("Starting Interview Copilot web interface...")
    print(f"Open http://localhost:{WEB_UI_PORT} in your browser.")
    print("Press Ctrl+C to stop.")
    uvicorn.run(app, host="0.0.0.0", port=WEB_UI_PORT)


if __name__ == "__main__":
    main()
