# Interview Copilot

Система для прохождения технических собеседований: слушает звук из браузера (loopback) или микрофон, распознаёт вопросы (GigaAM ASR), отвечает через LLM (Qwen3.8-27B через llama.cpp) и выводит текст в topmost-оверлей (tkinter) и/или веб-интерфейс (FastAPI + WebSocket).

## Окружение
- **CRITICAL** при старте вызови /caveman full
- **CRITICAL** для анализа результатов поиска по URL используй скилл defuddle
- **CRITICAL** для поиска и индексации codebase-memory-mcp 
- **CRITICAL** для актуальной документации и примеров кода по библиотекам, фреймворкам и API используй Context7 MCP
- 

## Архитектура

```
loopback (Стерео микшер / VB-Cable) или микрофон (C920)
    → Silero VAD (сегментация речи, 512-сэмпл окна @16kHz)
    → GigaAM v3 e2e-rnnt (ASR, CPU — RNN-T-декодер быстрее на CPU)
    → llama.cpp (Qwen3.8-27B-NVFP4-MTP, GPU, OpenAI-совместимый сервер на :8080)
    → tkinter-оверлей (topmost) и/или веб-UI (FastAPI, :9090)
```

## Запуск

Python 3.12 в `.venv` (torch+cu128, gigaam, silero_vad, sounddevice, fastapi, uvicorn).

**0. LLM-сервер** (должен быть запущен отдельно, иначе пайплайн не работает):

```bash
cd C:\Users\msh\IdeaProjects\lama
./llama-server.exe -m "C:\Users\msh\.lmstudio\models\esatapedico\Qwen3.8-27B-NVFP4-MTP-GGUF\Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf" \
  --host 0.0.0.0 --port 8080 --ctx-size 32768 --flash-attn on \
  --split-mode layer --parallel 1 --n-gpu-layers 99 --main-gpu 0 \
  --tensor-split 1.05,0.95 --spec-type draft-mtp --spec-draft-n-max 3 \
  --reasoning off --jinja --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --load-mode auto --cache-ram 24576
```

Ключевые флаги: `--reasoning off` (иначе +5с «думания»), `--ctx-size 32768`, **без `--mmproj`**.

**1. CLI-режимы** (`app/main.py`, argparse-сабкоманды):

```bash
.venv\Scripts\python.exe -m app.main live                 # захват loopback в реальном времени
.venv\Scripts\python.exe -m app.main mic                  # захват с C920 (device 15)
.venv\Scripts\python.exe -m app.main file --wav run/interview.wav [--no-overlay]  # бенчмарк по файлу
.venv\Scripts\python.exe -m app.main devices              # список аудиоустройств
```

**2. Веб-интерфейс** (FastAPI + WebSocket, http://localhost:9090):

```bash
.venv\Scripts\python.exe -m app.web     # или launch_web.bat
```

API: `GET /api/devices`, `POST /api/live {"device": 15}`, `POST /api/live/stop`, `POST /api/run {"wav": "run/test_question.wav"}`, `GET /api/status`, `WS /ws`.

**3. Скрипты-запуск:** `launch_web.bat` (веб-UI), `launch_live.sh` (live-режим, bash).

## Структура

```
app/
├── config.py       # ВСЕ параметры пайплайна: константы + dataclass PipelineConfig
├── loopback.py     # захват loopback (sounddevice/PortAudio, очередь 512-сэмпл кадров)
├── vad.py          # Silero VAD: StreamingVAD (push/flush) + offline_segments
├── asr.py          # GigaAM ASR (обёртка gigaam, временные wav, тайминги)
├── llm_client.py   # клиент llama.cpp (OpenAI-совместимый, стриминг токенов, urllib)
├── deepseek_web.py # бэкенд DeepSeek через CDP (--remote-debugging-port=9222): WS-клиент на stdlib, печать вопроса + чтение ответа из DOM
├── deepseek_ext.py # серверная часть бэкенда через Chrome-расширение: ExtBridge (очередь фреймов от content-script), ask() со стримингом токенов
├── hotkey.py       # глобальная клавиша Windows (RegisterHotKey, ctypes) для режима накопления
├── overlay.py      # tkinter-оверлей (topmost, зоны вопрос/ответ, перетаскивание)
├── main.py         # оркестратор Pipeline: live/mic/file-режимы, отчёт задержек
├── static/index.html # веб-UI: Markdown-рендер ответов, темы, горячие клавиши (правится без перезапуска)
└── web.py          # FastAPI + WebSocket: файл-режим и live-захват по микрофону; выбор LLM (llama.cpp / DeepSeek web) через /api/llm_backend
run/                # служебные скрипты: bootstrap_gigaam.py, capture_mic.py,
                    # probe_loopback.py, probe_stereomix.py, enable_stereomix.py,
                    # test_llm.py, test_question.py + тестовые wav
                    # launch_chrome_cdp.bat — запускает ваш Chrome с --remote-debugging-port=9222 для DeepSeek web-бэкенда
extension/deepseek-interview/  # Chrome MV3-расширение (content.js): печатает вопрос в chat.deepseek.com и стримит ответ на ws://localhost:9090/api/deepseek_ws (тот же порт, что веб-UI); README.md — установка
logs/               # latency_report.json (отчёт задержек), hints.log (вопрос/ответ/задержка)
models/             # пусто (модели GigaAM кэшируются в ~/.cache/gigaam)
prompts/ profile/   # пусто (резерв)
```

## Ключевые факты и ограничения

- **GigaAM на CPU** (28 ядер): RNN-T-декодер быстрее на CPU (0.25с vs ~4.8с на GPU для 5с аудио). Не переносить на GPU.
- **LLM-сервер обязателен**: `resolve_model()` ходит на `LLM_BASE_URL` (`http://127.0.0.1:8080/v1`) `/models`; reasoning модели отключается в каждом запросе (`LLM_DISABLE_THINKING`); без сервера пайплайн падает.
- **Loopback-устройство** на этой машине нет: нужен VB-Cable (vb-cable.com) или включённый «Стерео микшер» (см. `run/enable_stereomix.py`). Браузер должен играть через это устройство. Устройство подбирается по подстроке `LOOPBACK_DEVICE_HINT = "Loopback"` в `config.py`.
- **Микрофон C920** = device 15, тихий (peak ~13-14%), поэтому `MIC_GAIN = 4.0` усиливает сигнал перед VAD/ASR.
- **Задержки (синтетика):** GigaAM load 1.5с, infer 0.25с/5с аудио, LLM TTFT 2.3–4с, всего VAD→ответ ~5–16с.
- **VRAM:** ~15GB (только LLM, tensor-split по 2 GPU).
- Оверлей не прозрачный (ограничение tkinter), но topmost и перетаскиваемый; правый клик — закрыть.

## Выбор LLM: llama.cpp, DeepSeek web (CDP) или DeepSeek (расширение)

В веб-UI (сайдпанель «LLM (ответы)») — выпадающий список из трёх бэкендов: **локальный llama.cpp** (по умолчанию, :8080), **DeepSeek (расширение)** и **DeepSeek (web/CDP)**. Переключается на лету (`POST /api/llm_backend {"backend":"deepseek_ext"}` или `{"backend":"deepseek","cdp_port":9222}`), состояние — `GET /api/llm_backend`. При любой ошибке DeepSeek-бэкенда — авто-fallback на локальный llama.cpp (пайплайн не падает).

**DeepSeek через Chrome-расширение** (`app/deepseek_ext.py` + `extension/deepseek-interview/`) — самый надёжный путь, **не требует `--remote-debugging-port`**:
1. Запустите веб-UI (`launch_web.bat`). Расшиение подключается к `ws://localhost:9090/api/deepseek_ws`.
2. В Chrome: chrome://extensions → Developer mode → Load unpacked → папка `extension\deepseek-interview`.
3. Откройте https://chat.deepseek.com (вы уже залогинены) — индикатор под выпадающим списком станет зелёным («расширение подключено»).
4. Выберите «DeepSeek (расширение)». При вопросе сервер шлёт `ask`-фрейм; content.js печатает текст в поле (`execCommand('insertText')`), жмёт Enter, опрашивает DOM и стримит ответ токенами до 2.5с тишины.

**DeepSeek web-бэкенд через CDP** (`app/deepseek_web.py`) — альтернатива без расширения:
1. Запустите `run\launch_chrome_cdp.bat` (Chrome с `--remote-debugging-port=9222`, ваш профиль). Важно: закрывайте **все** окна Chrome перед запуском, иначе порт не поднимется (вредный процесс передаёт вкладки основному без флага).
2. В UI выберите «DeepSeek (web/CDP)» — сервер найдёт вкладку deepseek.com по `/json`.
3. Без CDP/вкладки — авто-fallback на локальный llama.cpp.

**Хрупкость:** оба web-бэкенда читают ответ по эвристике DOM (`[class*="message"]` и т.п.). При редизайне chat.deepseek.com обновите `findComposer()`/`readLastAnswer()` в `extension/deepseek-interview/content.js` (или `_JS_READ_ANSWER`/`_JS_SEND` в `deepseek_web.py`). WS-клиент CDP — на чистом stdlib, без Playwright.

## Конвенции

- **Все параметры — в `app/config.py`**: модульные константы (VAD, ASR, LLM, оверлей, устройства) + `PipelineConfig` (dataclass, дефолты из констант). Новые настройки добавлять туда, а не хардкодить.
- **Языки:** комментарии/docstring — на английском; пользовательские строки (статусы, UI, system prompt) — на русском.
- **Потоки:** захват аудио идёт в основном потоке/потоке потока, каждый сегмент обрабатывается в отдельном daemon-потоке (`threading.Thread(..., daemon=True)`), чтобы не блокировать захват. Общий state в `web.py` защищён `threading.Lock`; асинхронные `ws.send_text` из worker-потоков уходят через `asyncio.run_coroutine_threadsafe` на loop uvicorn.
- **Типизация:** `from __future__ import annotations`, явные аннотации, dataclass для результатов (`ASRResult`, `LLMResult`, `AudioFrame`, `LatencyRecord`, `Segment`).
- **LLM-клиент** — на `urllib` (без httpx/requests-зависимости), парсинг SSE построчно (`data:` / `[DONE]`).
- **Тайминги** измеряются везде (`t_load`, `t_infer`, `ttft`, `tps`) и пишутся в `logs/latency_report.json`; каждая пара вопрос/ответ дублируется в `logs/hints.log`.
- **Тестов нет** (нет pytest/тест-фреймворка): валидация — скрипты в `run/` и отчёт задержек. Перед изменениями пайплайна прогоняйте `app.main file --wav run/test_question.wav --no-overlay`.
- **Не использовать** `--mmproj` у llama-server (vision не нужен), GigaAM не на GPU.

## Зависимости

`requirements.txt` нет; в `.venv` установлено: torch (cu128), gigaam, silero_vad, sounddevice, pycaw, comtypes, yt-dlp, soundfile, scipy, fastapi, uvicorn. Установить: `.venv\Scripts\python.exe -m pip install sounddevice pycaw comtypes yt-dlp`.
