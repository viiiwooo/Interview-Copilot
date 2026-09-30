"""Central config for the interview copilot pipeline.

Pipeline: loopback -> Silero VAD -> GigaAM (ASR) -> llama.cpp (LLM) -> overlay window.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # interview-copilot/
RUN = ROOT / "run"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

# --- Audio / loopback -------------------------------------------------------
SAMPLE_RATE = 16000
# Windows loopback device label pattern (the browser tab's output goes here).
# Set to None to auto-pick the first device whose name contains LOOPBACK_DEVICE_HINT.
LOOPBACK_DEVICE_HINT = "Loopback"          # e.g. "Stereo Mix", "Loopback", "CABLE"
LOOPBACK_DEVICE = None                     # exact label, or None = auto by hint
# If the browser is on a specific virtual cable, put its exact name here.

# --- Mic capture (for testing / when loopback is unavailable) ----------------
# Device index of the C920 microphone (from `python -m app.main devices`).
MIC_DEVICE = 15                            # C920 (highest peak in testing)
# The C920 mic is quiet (peak ~13-14% at normal distance). Amplify the live
# capture so VAD and GigaAM see a normal-level signal. Clipped to int16 range.
MIC_GAIN = 4.0

# --- VAD (Silero) -----------------------------------------------------------
# The C920 mic is quiet (peak ~13-14% at normal speaking distance), so the VAD
# probability only barely crosses 0.5. A lower threshold makes speech reliable
# without false positives (silence sits around 0.08, well below 0.3).
VAD_THRESHOLD = 0.3
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 300        # gap that ends a speech segment
VAD_MAX_SILENCE_MS = 1200       # hard stop if silence runs this long mid-speech
VAD_MAX_SPEECH_S = 22.0         # GigaAM .transcribe is limited to ~25s; keep below
# Live transcript: while a phrase is still being spoken, re-recognize it every
# PARTIAL_ASR_MS and show the text in the UI (0 = off). Skipped when the ASR is
# busy with a final segment, so it never delays an answer.
PARTIAL_ASR_MS = 1000

# --- ASR (GigaAM) -----------------------------------------------------------
GIGAM_MODEL = "v3_e2e_rnnt"     # punctuation + casing + ITN, lowest WER
# NOTE: on this box the RNN-T decoder is FASTER on CPU than GPU (0.25s vs ~4.8s
# for a 5s clip) because the decoder loop doesn't parallelize well on the GPU.
# Keep GigaAM on CPU (28 cores) and leave the GPUs for the LLM.
GIGAM_DEVICE = "cpu"            # "cuda" or "cpu"
GIGAM_FP16_ENCODER = False

# --- LLM (llama.cpp server) -------------------------------------------------
# OpenAI-compatible base URL (with or without the trailing /v1). Use 127.0.0.1,
# not localhost: on Windows localhost resolves to ::1 first and adds ~200ms per request.
LLM_BASE_URL = "http://127.0.0.1:8080/v1"
# The served model emits reasoning_content before the answer; with a small
# max_tokens the whole budget goes to thinking and the answer comes back empty.
# Disable it per request (chat_template_kwargs + reasoning_effort).
LLM_DISABLE_THINKING = True
# Answer length cap for the local LLM, in tokens. Measured ~2.2 Russian chars
# per token (220 tokens cut answers at ~478 chars), so 1400 tokens ≈ 3000 chars.
LLM_MAX_TOKENS = 1400
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9
# System prompt: concise, interview-oriented, Russian.
LLM_SYSTEM_PROMPT = (
    "Ты — ассистент кандидата на техническом собеседовании. "
    "Тебе передают вопрос интервьюера (распознанный с голоса). "
    "Английские термины и аббревиатуры записаны кириллицей на слух и могут быть "
    "искажены (например, «кубернетес», «эс кью эль», «джи ар пи си») — "
    "восстанови их по контексту и в ответе пиши в оригинальном написании. "
    "Дай короткий, структурированный ответ по делу (3-6 пунктов или абзац), "
    "на русском, без вступлений и воды. Если вопрос не ясен — уточни в одну строку. "
    "Не выдумывай факты о себе кандидате; опирайся на общий технический опыт."
)

# --- Web UI / DeepSeek web backend (Chrome extension) ------------------------
# The extension lives in the user's own Chrome on chat.deepseek.com. It types the
# question into the composer and reads the assistant reply from the DOM, then
# streams it back to the server over a WebSocket at DEEPSEEK_EXT_WS_PORT — the
# SAME port as the web UI (launch_web.bat -> :9090), so one uvicorn serves both.
WEB_UI_PORT = 9090                   # http://localhost:9090
DEEPSEEK_EXT_WS_PORT = WEB_UI_PORT   # ws://localhost:9090/api/deepseek_ws
DEEPSEEK_EXT_TIMEOUT_S = 120.0       # give up waiting for an answer after this
# Message sent to DeepSeek for each recognized question. The web chat has no
# system role, so the system prompt (editable in the UI) goes in front of the
# question; with an empty prompt only the recognized text is sent.
DEEPSEEK_PROMPT_TEMPLATE = "{system}\n\n{question}"


# --- Accumulate mode (web UI) ------------------------------------------------
# When on, recognized speech is collected instead of being answered segment by
# segment; the collected text goes to the LLM as one question on a hotkey.
# The hotkey is system-wide (works while the interview window has focus).
ACCUMULATE_DEFAULT = False
SEND_HOTKEY = "F8"                   # e.g. "F9", "Pause", "Ctrl+Space"


# --- Meeting recording + review (web UI) --------------------------------------
SESSIONS = ROOT / "sessions"         # one folder per recorded meeting
RECORD_DEFAULT = False               # "Записывать встречу" checkbox at start
RECORD_AUDIO = True                  # also save audio.wav next to transcript.md
REVIEW_MAX_TOKENS = 2500             # review length cap (local LLM)
# Long meetings are cut to the LAST this many chars so the prompt fits the
# local model's context (32k tokens ≈ 70k chars; leave room for the answer).
REVIEW_MAX_CHARS = 40000
REVIEW_PROMPT = (
    "Ты — опытный интервьюер и карьерный коуч. Ниже стенограмма технического "
    "собеседования, распознанная с голоса (возможны ошибки распознавания). В ней "
    "реплики интервьюера, иногда реплики кандидата, и подсказки, которые "
    "кандидату показывал ассистент.\n"
    "Сделай разбор в Markdown на русском:\n"
    "## Кратко — о чём было собеседование, роль и стек, если понятно.\n"
    "## Вопросы — таблица: вопрос | суть | ключевые пункты сильного ответа.\n"
    "## Слабые места — темы, где кандидату стоит подтянуться, с конкретикой.\n"
    "## Что повторить — короткий чек-лист для подготовки к следующему этапу.\n"
    "## Вопросы работодателю — что стоило уточнить у интервьюера.\n"
    "Подсказки ассистента — рабочий инструмент кандидата: не оценивай сам факт их "
    "использования, бери их как материал и дополняй, если они неполные.\n"
    "Пиши по делу, без воды. Не выдумывай то, чего нет в стенограмме."
)


# --- Overlay window ---------------------------------------------------------
OVERLAY_WIDTH = 760
OVERLAY_HEIGHT = 520
OVERLAY_X = 90
OVERLAY_Y = 90

# --- Latency measurement ----------------------------------------------------
LOG_TIMINGS = True


@dataclass
class PipelineConfig:
    # audio
    sample_rate: int = SAMPLE_RATE
    loopback_device_hint: str = LOOPBACK_DEVICE_HINT
    loopback_device: str | None = LOOPBACK_DEVICE
    mic_device: int = MIC_DEVICE            # C920 mic device index
    mic_gain: float = MIC_GAIN             # amplify the quiet mic before VAD/ASR
    # vad
    vad_threshold: float = VAD_THRESHOLD
    vad_min_speech_ms: int = VAD_MIN_SPEECH_MS
    vad_min_silence_ms: int = VAD_MIN_SILENCE_MS
    vad_max_silence_ms: int = VAD_MAX_SILENCE_MS
    vad_max_speech_s: float = VAD_MAX_SPEECH_S
    partial_asr_ms: int = PARTIAL_ASR_MS
    # asr
    gigaam_model: str = GIGAM_MODEL
    gigaam_device: str = GIGAM_DEVICE
    gigaam_fp16_encoder: bool = GIGAM_FP16_ENCODER
    # llm
    llm_base_url: str = LLM_BASE_URL
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_temperature: float = LLM_TEMPERATURE
    llm_top_p: float = LLM_TOP_P
    llm_system_prompt: str = LLM_SYSTEM_PROMPT
    # deepseek web backend (chrome extension)
    deepseek_ext_ws_port: int = DEEPSEEK_EXT_WS_PORT
    deepseek_ext_timeout_s: float = DEEPSEEK_EXT_TIMEOUT_S
    deepseek_prompt_template: str = DEEPSEEK_PROMPT_TEMPLATE
    # overlay
    overlay_width: int = OVERLAY_WIDTH
    overlay_height: int = OVERLAY_HEIGHT
    overlay_x: int = OVERLAY_X
    overlay_y: int = OVERLAY_Y
    # timing
    log_timings: bool = LOG_TIMINGS
    # file-based test input (for the YouTube-recording benchmark): a wav path.
    # When set, the pipeline reads this file instead of the live loopback device.
    test_wav: str | None = None
