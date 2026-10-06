# Interview Copilot 🎧

Подсказчик для технических собеседований. Слушает звук созвона (вкладку браузера
или микрофон), распознаёт вопросы интервьюера (GigaAM), отвечает через LLM
(локальный сервер или DeepSeek в вашем Chrome) и показывает ответ в веб-интерфейсе,
оформленный для быстрого чтения.

## Архитектура

```
вкладка созвона (getDisplayMedia) / микрофон     ← браузер, Web Audio, 16 кГц
    → WebSocket /ws (PCM-кадры по 512 сэмплов)
    → Silero VAD (сегментация речи)
    → GigaAM v3 e2e-rnnt (ASR, CPU, нормализация громкости)
    → [режим накопления: текст копится до нажатия клавиши]
    → LLM: локальный OpenAI-совместимый сервер (127.0.0.1:8080/v1)
           или DeepSeek через Chrome-расширение / CDP
    → веб-интерфейс http://localhost:9090 (Markdown, стриминг)
```

CLI-режим (`app.main`) с tkinter-оверлеем и захватом loopback/микрофона на сервере
тоже остался, но основной сценарий — веб-интерфейс.

## Установка

Нужны Windows 10/11, Python 3.10+ (проверено на 3.12), Chrome и `ffmpeg` в `PATH`.

### 1. Окружение и зависимости

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip

# PyTorch. GigaAM у нас работает на CPU, но колёса с CUDA тоже подходят
# (проверено: torch 2.10.0+cu128). Только CPU:
.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# или с CUDA 12.8:
# .venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# остальное
.venv\Scripts\python.exe -m pip install fastapi uvicorn[standard] pydantic numpy soundfile sounddevice silero-vad
.venv\Scripts\python.exe -m pip install scipy   # нужен только CLI-режиму app.main
```

### 2. GigaAM (распознавание речи)

[GigaAM](https://github.com/salute-developers/GigaAM) — открытая ASR-модель Сбера
для русского языка. Пакета на PyPI нет, ставится из GitHub:

```bash
.venv\Scripts\python.exe -m pip install "gigaam @ git+https://github.com/salute-developers/GigaAM"
```

Нужен установленный `git`. Проверенная версия — `gigaam 0.2.0`, коммит `7447938`;
чтобы поставить именно её, допишите в конец URL `@7447938d791c4f3e643386ee22c33777004293a5`.

**Веса.** Скачивать вручную не нужно: при первом запуске `gigaam.load_model("v3_e2e_rnnt")`
сам загружает их с CDN Сбера (`https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/`)
в `%USERPROFILE%\.cache\gigaam\`:

| Файл | Размер |
|------|--------|
| `v3_e2e_rnnt.ckpt` | ~430 МБ |
| `v3_e2e_rnnt_tokenizer.model` | ~250 КБ |

Скачать заранее (или проверить, что всё работает):

```bash
.venv\Scripts\python.exe -c "import gigaam; gigaam.load_model('v3_e2e_rnnt', device='cpu', fp16_encoder=False)"
```

На машине без интернета скопируйте оба файла из `.cache\gigaam` другой машины
в ту же папку. Модель выбирается в `app/config.py`: `GIGAM_MODEL = "v3_e2e_rnnt"`
(с пунктуацией, регистром и нормализацией чисел), `GIGAM_DEVICE = "cpu"`.

**ffmpeg.** GigaAM читает аудио через `ffmpeg`, без него распознавание падает
с `Failed to load audio`. Установка:

```bash
winget install Gyan.FFmpeg
```

После установки откройте новый терминал и проверьте: `ffmpeg -version`.

### 3. Silero VAD

Ставится пакетом `silero-vad` (шаг 1), веса лежат внутри пакета, отдельно ничего
качать не нужно.

## Быстрый старт

1. **Запустите LLM-сервер** на `http://127.0.0.1:8080/v1` — любой OpenAI-совместимый
   (llama.cpp, LM Studio и т. п.). Пример для llama.cpp:

   ```bash
   llama-server.exe -m <модель>.gguf --host 127.0.0.1 --port 8080 \
     --ctx-size 32768 --flash-attn on --n-gpu-layers 99 --jinja
   ```

   «Думание» (reasoning) модели отключается в каждом запросе
   (`chat_template_kwargs.enable_thinking=false`, `reasoning_effort=none`), иначе
   ответ приходит пустым или с задержкой. Проверка сервера:

   ```bash
   .venv\Scripts\python.exe run\test_llm.py
   ```

2. **Запустите веб-интерфейс:**

   ```bash
   launch_web.bat        # или .venv\Scripts\python.exe -m uvicorn app.web:app --port 9090
   launch_web_gpu.bat    # то же, но GigaAM на GPU (GIGAAM_DEVICE=cuda)
   ```

   Откройте http://localhost:9090 в Chrome, лучше в отдельном окне у верхнего края
   экрана, ближе к камере.

3. **Начните захват:** панель справа → «Захват звука» → **🖥 Вкладка** → **▶ Захват** →
   выберите вкладку с созвоном и включите **«Также поделиться звуком вкладки»**.
   Для Zoom/Teams-приложения: «Весь экран» + «Поделиться системным звуком».

После изменения Python-кода сервер нужно перезапустить. Правки
`app/static/index.html` видны после перезагрузки страницы.

## Веб-интерфейс

**Живой текст вопроса**
- Пока интервьюер говорит, фраза распознаётся раз в ~1 с и показывается сверху
  бледной строкой с красной точкой. В конце фразы её заменяет финальный вопрос.
- Живое распознавание не задерживает ответы: если GigaAM занят финальной фразой,
  промежуточный кусок пропускается. Отключается `PARTIAL_ASR_MS = 0`.

**Лента ответов**
- Новый вопрос появляется **сверху**, под камерой; старые уходят вниз и приглушаются.
- Вопрос — янтарный курсив с полосой слева, ответ — обычный текст.
- Ответ рендерится из Markdown:
  - **жирный** — голубым с подложкой;
  - *курсив* — янтарным;
  - `код` — оранжевым;
  - у пунктов цветные номера и значки (●, ◆, –);
  - «Термин — пояснение» в начале пункта выделяется автоматически.
- Если вы пролистали вниз, появляется кнопка «↑ К последнему ответу».

**Захват звука**
- Два источника с отдельными настройками:
  - **Вкладка** — усиление ×1, пауза конца фразы 800 мс, авто-усиление включено;
  - **Микрофон** — усиление ×4, пауза 300 мс.
  - **Оба** — вкладка (собеседник) + микрофон (вы) смешиваются в один поток;
    отдельный слайдер «Микрофон» (×4) задаёт громкость микрофона в миксе.
    Нужны наушники: эхоподавление выключено, иначе микрофон повторно ловит собеседника.
    Ваши реплики тоже распознаются и уходят в LLM как вопросы — удобно с режимом накопления.
- Полоска уровня показывает, что звук идёт. Слайдеры: усиление, порог VAD, пауза конца фразы.

**Режим накопления** (карточка «Отправка вопросов»)
- Галочка «Копить речь, отправлять по клавише»: распознанные фразы не уходят в LLM
  по одной, а копятся в карточке сверху ленты. Текст можно отредактировать.
- Отправка одним вопросом:
  - **глобальной клавишей** (F8, F9, F10, F12, ~, Pause, Ctrl+Space, Ctrl+Enter) —
    работает из любого окна, даже когда в фокусе созвон;
  - в окне копилота — **Пробел**, **Enter** или кнопка «➤ Отправить».
- **Esc** — очистить накопленное.
- Пока режим включён, выбранная клавиша перехватывается у всех программ (для `~`
  это значит, что нельзя напечатать `ё`).

**Запись встречи и разбор** (карточка «Запись встречи»)
- Галочка «Записывать встречу»: пока идёт захват, в `sessions/<дата_время>/` пишутся
  `transcript.md` (каждая распознанная реплика с временем + подсказки LLM) и,
  если отмечено «Сохранять аудио», `audio.wav`. Сверху горит значок **● REC**.
- «📝 Разобрать» отправляет стенограмму выбранной встречи в текущий LLM и сохраняет
  результат в `review.md` (стрим в окне просмотра). Кнопки «Стенограмма» / «Разбор»
  открывают файлы прямо в интерфейсе.
- Промпт разбора настраивается в «Промпт разбора встречи» (по умолчанию: кратко,
  таблица вопросов, слабые места, что повторить, вопросы работодателю).
- При захвате вкладки записывается только собеседник; ваш голос попадает в запись,
  если источник — микрофон или «Оба».

**LLM (ответы)**: локальный LLM, DeepSeek (расширение) или DeepSeek (web/CDP).
При ошибке DeepSeek ответ автоматически берётся у локального LLM.

**Горячие клавиши** (когда фокус в окне копилота)

| Клавиша | Действие |
|---------|----------|
| `+` / `−` | размер шрифта ответа (по умолчанию 15 px, сохраняется) |
| `H` | скрыть/показать панель настроек |
| `T` | светлая/тёмная тема |
| `Home` / `End` | к последнему ответу |
| `C` | копировать последний ответ (Markdown) |
| `Пробел` / `Enter` / `Esc` | в режиме накопления: отправить / отправить / очистить |

Клавиши работают и в русской раскладке.

## DeepSeek через Chrome-расширение

Ответы от chat.deepseek.com в вашем обычном Chrome, без API-ключа и без
`--remote-debugging-port`.

1. Запустите `launch_web.bat`.
2. `chrome://extensions` → Developer mode → **Load unpacked** →
   `extension\deepseek-interview`.
3. Откройте https://chat.deepseek.com (вы залогинены). Для скорости выключите
   **DeepThink** и **Search**.
4. В интерфейсе: «LLM (ответы)» → **DeepSeek (расширение)**. Под списком должно
   появиться «подключено v1.2.0». Если там «расширение устарело», нажмите «Обновить»
   у расширения и перезагрузите вкладку DeepSeek.

Как это устроено:
- расширение читает **только новые блоки ответа** (`.ds-markdown`, без блока
  размышлений), поэтому не путает вопрос и ответ;
- переводит ответ в чистый Markdown (`md.js`): без кнопок «Копировать»/«Скачать»,
  номера и значки пунктов сохраняются;
- конец ответа определяется по закрытию потока DeepSeek (`page_hook.js`), а не по
  таймауту тишины.

Подробности — в [extension/deepseek-interview/README.md](extension/deepseek-interview/README.md).
В DeepSeek отправляется системный промпт из UI и распознанный вопрос (`DEEPSEEK_PROMPT_TEMPLATE`); при пустом промпте — только вопрос.

## CLI-режимы

```bash
.venv\Scripts\python.exe -m app.main live      # loopback-устройство сервера + tkinter-оверлей
.venv\Scripts\python.exe -m app.main mic       # микрофон (device 15) + оверлей
.venv\Scripts\python.exe -m app.main file --wav test.wav --no-overlay   # прогон WAV
.venv\Scripts\python.exe -m app.main devices   # список аудиоустройств
```

Отчёт задержек пишется в `logs/latency_report.json`, пары вопрос/ответ — в
`logs/hints.log`. Для `live` нужно loopback-устройство («Стерео микшер» или
VB-Cable); веб-захват вкладки его не требует.

## API веб-сервера

| Метод | Путь | Назначение |
|-------|------|-----------|
| GET | `/` | веб-интерфейс (`app/static/index.html`) |
| WS | `/ws` | обновления UI + PCM-кадры захвата из браузера |
| POST | `/api/run` `{"wav": "..."}` | прогон WAV-файла |
| GET | `/api/status` | текущие вопрос/ответ/статус |
| GET/POST | `/api/prompt` | системный промпт локального LLM |
| POST | `/api/vad_config` `{"min_silence_ms", "threshold"}` | параметры VAD |
| GET/POST | `/api/llm_backend` `{"backend": "local"\|"deepseek_ext"\|"deepseek"}` | выбор LLM |
| GET/POST | `/api/accumulate` `{"enabled", "hotkey"}` | режим накопления |
| POST | `/api/pending`, `/api/pending/send`, `/api/pending/clear` | накопленный текст |
| GET/POST | `/api/record` `{"enabled", "audio"}` | запись встречи |
| GET | `/api/sessions`, `/api/sessions/{name}/transcript\|review` | записанные встречи |
| POST | `/api/review` `{"session"}` | разбор встречи → `review.md` |
| GET/POST | `/api/review_prompt` | промпт разбора |
| WS | `/api/deepseek_ws` | канал Chrome-расширения DeepSeek |
| GET | `/api/deepseek_ext` | статус расширения |
| GET/POST | `/api/devices`, `/api/live`, `/api/live/stop` | захват микрофона на сервере |

## Настройки (`app/config.py`)

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | LLM-сервер (с `/v1` или без) |
| `LLM_MAX_TOKENS` | 1400 | ≈ 3000 символов ответа |
| `LLM_DISABLE_THINKING` | True | отключить reasoning модели |
| `LLM_SYSTEM_PROMPT` | … | промпт локального LLM (меняется и в UI) |
| `DEEPSEEK_PROMPT_TEMPLATE` | … | сообщение для DeepSeek: системный промпт + вопрос |
| `VAD_THRESHOLD` / `VAD_MIN_SILENCE_MS` | 0.3 / 300 | чувствительность VAD и пауза конца фразы |
| `PARTIAL_ASR_MS` | 1000 | период живого распознавания (0 — выкл.) |
| `GIGAM_DEVICE` | `cpu` | устройство GigaAM: `cpu` / `cuda` (env `GIGAAM_DEVICE`, переключатель в UI) |
| `ACCUMULATE_DEFAULT` / `SEND_HOTKEY` | False / `F8` | режим накопления при старте |
| `WEB_UI_PORT` | 9090 | порт веб-интерфейса и WS расширения |
| `RECORD_DEFAULT` / `RECORD_AUDIO` | False / True | запись встречи при старте, сохранять аудио |
| `REVIEW_PROMPT` / `REVIEW_MAX_TOKENS` | … / 2500 | промпт и длина разбора |
| `REVIEW_MAX_CHARS` | 40000 | длинная стенограмма обрезается до последних N символов |

## Структура

```
app/
├── config.py        # все параметры
├── web.py           # FastAPI + WebSocket: захват из браузера, LLM, режим накопления
├── static/index.html# веб-интерфейс (Markdown-рендер, темы, горячие клавиши)
├── asr.py           # GigaAM + нормализация громкости сегмента
├── vad.py           # Silero VAD (streaming + offline)
├── llm_client.py    # OpenAI-совместимый клиент (стриминг, без reasoning)
├── deepseek_ext.py  # мост к Chrome-расширению (снимки текста, версия, пинги)
├── deepseek_web.py  # DeepSeek через CDP (использует тот же md.js)
├── hotkey.py        # глобальная клавиша Windows (RegisterHotKey)
├── recorder.py      # запись встречи: transcript.md + audio.wav по сессиям
├── main.py          # CLI-режимы: live / mic / file
├── loopback.py      # захват loopback на сервере (sounddevice)
└── overlay.py       # tkinter-оверлей (Markdown → простой текст)
extension/deepseek-interview/   # Chrome MV3: content.js, md.js, page_hook.js
run/                 # служебные скрипты: test_llm.py, probe_*.py, launch_chrome_cdp.bat
logs/                # latency_report.json, hints.log
sessions/            # записанные встречи: transcript.md, audio.wav, review.md
```

## Замеры (локальный LLM, `test.wav`)

| Стадия | Время |
|--------|-------|
| Загрузка GigaAM | ~1.7 с (один раз) |
| ASR фразы ~6 с | ~0.2–1 с (CPU) |
| LLM до первого токена | ~0.9–1.0 с |
| Генерация | ~45 ток/с |
| От конца речи до полного ответа | ~4 с |

## Известные ограничения

- Отладочные клипы пишутся в `run/`: `browser_cap_*.wav` при остановке захвата,
  `empty_seg_*.wav` для пустых сегментов. Их можно удалять.
- Сервер слушает `0.0.0.0` без авторизации, WebSocket не проверяет `Origin`.
  Используйте только в доверенной сети.
- DeepSeek-бэкенды опираются на вёрстку chat.deepseek.com. При редизайне правьте
  селекторы в `content.js` / `md.js`.
- Глобальная клавиша работает только в Windows.
- Тестов нет. Проверка — `run/test_llm.py` и
  `app.main file --wav test.wav --no-overlay`.
