"""Interview copilot — main orchestrator.

Pipeline:  loopback -> Silero VAD -> GigaAM (ASR) -> llama.cpp (LLM) -> overlay

Two run modes:
  * live       -- capture the browser's loopback device in real time.
  * file test  -- read a WAV (e.g. a YouTube interview recording), run VAD over
                  it, transcribe each speech segment, ask the LLM, and print a
                  per-segment latency report. This is the benchmark mode.

Usage:
  python -m app.main live
  python -m app.main file --wav run/interview.wav
  python -m app.main file --wav run/interview.wav --no-overlay
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import PipelineConfig, RUN, LOGS
from .loopback import LoopbackCapture, list_devices
from .vad import StreamingVAD, offline_segments, Segment
from .asr import GigaAMASR
from .llm_client import ask, resolve_model
from .overlay import Overlay


@dataclass
class LatencyRecord:
    seg_start: float
    seg_end: float
    t_vad_done: float
    t_asr_done: float
    t_llm_first: float
    t_llm_done: float
    text: str
    answer: str

    @property
    def asr_s(self) -> float:
        return self.t_asr_done - self.t_vad_done

    @property
    def llm_ttft_s(self) -> float:
        return self.t_llm_first - self.t_asr_done

    @property
    def total_s(self) -> float:
        return self.t_llm_done - self.t_vad_done


def _load_wav_int16(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import scipy.signal as sps
        data = sps.resample_poly(data, 16000, sr)
        sr = 16000
    return (np.clip(data, -1, 1) * 32767).astype(np.int16), sr


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.overlay = Overlay(cfg.overlay_width, cfg.overlay_height,
                               cfg.overlay_x, cfg.overlay_y)
        self.records: list[LatencyRecord] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.llm_model = ""

    # ---- stage setup ------------------------------------------------------
    def setup(self) -> None:
        print(f"[pipeline] loading GigaAM {self.cfg.gigaam_model} on {self.cfg.gigaam_device} ...",
              flush=True)
        t0 = time.time()
        self.asr = GigaAMASR(self.cfg.gigaam_model, self.cfg.gigaam_device,
                             self.cfg.gigaam_fp16_encoder)
        print(f"[pipeline] GigaAM ready in {time.time()-t0:.1f}s (load={self.asr.t_load:.1f}s)",
              flush=True)
        self.vad = StreamingVAD(
            self.cfg.sample_rate, self.cfg.vad_threshold,
            self.cfg.vad_min_speech_ms, self.cfg.vad_min_silence_ms,
            self.cfg.vad_max_silence_ms, self.cfg.vad_max_speech_s,
        )
        self.llm_model = resolve_model(self.cfg.llm_base_url)
        print(f"[pipeline] LLM model: {Path(self.llm_model).name}", flush=True)

    # ---- one speech segment -> ASR -> LLM -> overlay ----------------------
    def process_segment(self, seg: Segment, show_overlay: bool = True) -> LatencyRecord:
        if show_overlay:
            self.overlay.set_status(f"речь {seg.start:.1f}-{seg.end:.1f}s → распознаю...")
        t0 = time.time()
        r = self.asr.transcribe(seg.audio, self.cfg.sample_rate)
        t_asr = time.time()
        text = r.text.strip()
        print(f"[asr] {seg.start:6.1f}-{seg.end:6.1f}s ({r.t_infer:.2f}s): {text!r}", flush=True)

        if not text:
            # noise/empty segment: keep the previous question and answer
            if show_overlay:
                self.overlay.set_status(
                    f"сегмент {seg.start:.1f}-{seg.end:.1f}s пустой (шум) — пропущен")
            rec = LatencyRecord(seg.start, seg.end, seg.t_vad_done, t_asr, t_asr, t_asr,
                                text, "")
            self._record(rec)
            return rec

        if show_overlay:
            self.overlay.set_question(text)
            self.overlay.set_status(f"вопрос готов ({r.t_infer:.1f}s) → отвечаю...")
            self.overlay.set_answer("")

        # LLM streaming; start a timer for first token
        first = {"t": None}
        def on_token(tok: str):
            if first["t"] is None:
                first["t"] = time.time()
                if show_overlay:
                    self.overlay.set_status("ответ идёт...")
            if show_overlay:
                self.overlay.set_answer(tok, append=True)

        t_llm0 = time.time()
        llm = ask(self.cfg.llm_base_url, self.llm_model,
                  self.cfg.llm_system_prompt, text,
                  self.cfg.llm_max_tokens, self.cfg.llm_temperature,
                  self.cfg.llm_top_p, on_token)  # always: it records the first-token time
        t_llm_done = time.time()
        t_llm_first = first["t"] or t_llm_done
        print(f"[llm] ttft={llm.ttft:.2f}s total={llm.total:.2f}s tps={llm.tps:.1f} "
              f"({llm.n_tokens} tok)", flush=True)
        if show_overlay:
            self.overlay.set_status(
                f"готово · ASR {r.t_infer:.1f}s · LLM ttft {llm.ttft:.1f}s · total {llm.total:.1f}s")

        rec = LatencyRecord(seg.start, seg.end, seg.t_vad_done, t_asr,
                            t_llm_first, t_llm_done, text, llm.text)
        self._record(rec)
        if show_overlay:
            lat = rec
            self.overlay.set_latency(
                f"VAD→ASR {lat.asr_s:.2f}s | ASR→LLM1 {lat.llm_ttft_s:.2f}s | "
                f"всего {lat.total_s:.2f}s")
        return rec

    def _record(self, rec: LatencyRecord) -> None:
        with self._lock:
            self.records.append(rec)
        # Append to a human-readable hints log (question + answer + latency)
        self._append_hint_log(rec)

    def _append_hint_log(self, rec: LatencyRecord) -> None:
        """Append a hint (question + answer + latency) to logs/hints.log."""
        if not rec.text:
            return
        LOGS.mkdir(parents=True, exist_ok=True)
        log = LOGS / "hints.log"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{ts}] {rec.seg_start:.1f}-{rec.seg_end:.1f}s\n")
            f.write(f"ВОПРОС: {rec.text}\n")
            f.write(f"ОТВЕТ: {rec.answer}\n")
            f.write(f"Задержка: ASR {rec.asr_s:.2f}s | LLM TTFT {rec.llm_ttft_s:.2f}s | всего {rec.total_s:.2f}s\n")
        print(f"[hint] logged -> {log}", flush=True)

    # ---- live mode --------------------------------------------------------
    def run_live(self) -> None:
        self.setup()
        self.overlay.set_status("запуск захвата loopback...")
        cap = LoopbackCapture(self.cfg.sample_rate, self.cfg.loopback_device_hint,
                              self.cfg.loopback_device)
        cap.start()
        print(f"[pipeline] capturing from device {cap.device_index}: {cap.device_name}",
              flush=True)
        self.overlay.set_status(f"слушаю: {cap.device_name}")

        def on_segment(seg: Segment):
            # process in a worker so audio capture keeps flowing
            threading.Thread(target=self.process_segment, args=(seg,), daemon=True).start()

        print("[pipeline] live loop running; Ctrl+C to stop", flush=True)
        try:
            while not self._stop.is_set():
                frame = cap.read(timeout=0.2)
                if frame is None:
                    continue
                self.vad.push(frame.samples, frame.t_start, on_segment)
        except KeyboardInterrupt:
            print("\n[pipeline] stopping", flush=True)
        finally:
            self.vad.flush(on_segment)
            cap.stop()
            self.report()

    # ---- file/benchmark mode ---------------------------------------------
    def run_file(self, wav: str, show_overlay: bool = True) -> None:
        self.setup()
        print(f"[pipeline] loading {wav} ...", flush=True)
        audio, sr = _load_wav_int16(wav)
        print(f"[pipeline] {len(audio)/sr:.1f}s of audio @ {sr}Hz", flush=True)
        if show_overlay:
            self.overlay.set_status(f"файл {Path(wav).name}: ищу речь...")

        t0 = time.time()
        segs = offline_segments(audio, sr, self.cfg.vad_threshold,
                                self.cfg.vad_min_speech_ms, self.cfg.vad_min_silence_ms)
        print(f"[pipeline] VAD found {len(segs)} speech segments in {time.time()-t0:.1f}s",
              flush=True)
        for i, seg in enumerate(segs):
            print(f"\n=== segment {i+1}/{len(segs)} ===", flush=True)
            self.process_segment(seg, show_overlay=show_overlay)
            if show_overlay:
                time.sleep(0.2)
        self.report()

    def report(self) -> None:
        if not self.records:
            print("[report] no segments processed", flush=True)
            return
        recs = [r for r in self.records if r.text]
        if not recs:
            print("[report] no non-empty transcriptions", flush=True)
            return
        asr = [r.asr_s for r in recs]
        ttft = [r.llm_ttft_s for r in recs]
        tot = [r.total_s for r in recs]
        print("\n" + "=" * 56, flush=True)
        print("LATENCY REPORT", flush=True)
        print("=" * 56, flush=True)
        print(f"segments with text : {len(recs)}", flush=True)
        print(f"ASR (VAD→text)     : mean {np.mean(asr):.2f}s  max {np.max(asr):.2f}s", flush=True)
        print(f"LLM TTFT (→1 tok)  : mean {np.mean(ttft):.2f}s  max {np.max(ttft):.2f}s", flush=True)
        print(f"total (VAD→done)   : mean {np.mean(tot):.2f}s  max {np.max(tot):.2f}s", flush=True)
        # save JSON
        LOGS.mkdir(parents=True, exist_ok=True)
        out = LOGS / "latency_report.json"
        data = {
            "summary": {
                "n": len(recs),
                "asr_mean": round(float(np.mean(asr)), 3),
                "asr_max": round(float(np.max(asr)), 3),
                "llm_ttft_mean": round(float(np.mean(ttft)), 3),
                "llm_ttft_max": round(float(np.max(ttft)), 3),
                "total_mean": round(float(np.mean(tot)), 3),
                "total_max": round(float(np.max(tot)), 3),
            },
            "records": [
                {"start": r.seg_start, "end": r.seg_end, "asr_s": round(r.asr_s, 3),
                 "llm_ttft_s": round(r.llm_ttft_s, 3), "total_s": round(r.total_s, 3),
                 "text": r.text, "answer": r.answer[:400]}
                for r in recs
            ],
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] saved -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interview copilot pipeline")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("live", help="capture loopback device in real time")
    sub.add_parser("mic", help="capture from C920 microphone in real time")
    f = sub.add_parser("file", help="run over a WAV file (benchmark)")
    f.add_argument("--wav", required=True, help="path to a wav file")
    f.add_argument("--no-overlay", action="store_true", help="headless, no window")
    sub.add_parser("devices", help="list audio devices and exit")
    args = ap.parse_args()

    cfg = PipelineConfig()

    if args.mode == "devices":
        print(list_devices())
        return

    if args.mode == "live":
        pipe = Pipeline(cfg)
        _run_live_with_overlay(pipe)
        return

    if args.mode == "mic":
        pipe = Pipeline(cfg)
        _run_mic_with_overlay(pipe, cfg.mic_device)
        return

    if args.mode == "file":
        pipe = Pipeline(cfg)
        show = not args.no_overlay
        if show:
            # overlay mainloop on main thread; pipeline on a worker
            pipe.overlay.build()
            threading.Thread(target=pipe.run_file, args=(args.wav, True), daemon=True).start()
            try:
                pipe.overlay.root.mainloop()
            except KeyboardInterrupt:
                pass
        else:
            pipe.run_file(args.wav, show_overlay=False)


def _run_live_with_overlay(pipe: Pipeline) -> None:
    pipe.overlay.build()
    threading.Thread(target=pipe.run_live, daemon=True).start()
    try:
        pipe.overlay.root.mainloop()
    except KeyboardInterrupt:
        pipe._stop.set()


def _run_mic_with_overlay(pipe: Pipeline, mic_device: int) -> None:
    """Run the pipeline with mic capture (C920) in real time."""
    import sounddevice as sd
    import numpy as np

    pipe.overlay.build()

    def mic_capture_loop():
        """Capture from the mic and push frames to VAD."""
        pipe.setup()
        pipe.overlay.set_status(f"слушаю микрофон (device {mic_device})...")
        print(f"[mic] capturing from device {mic_device} (C920)", flush=True)

        def on_segment(seg):
            threading.Thread(target=pipe.process_segment, args=(seg,), daemon=True).start()

        # Use a stream for real-time capture
        with sd.InputStream(
            samplerate=pipe.cfg.sample_rate,
            channels=1,
            dtype="int16",
            device=mic_device,
            blocksize=512,
        ) as stream:
            print("[mic] stream started; Ctrl+C to stop", flush=True)
            t_start = time.time()
            try:
                while not pipe._stop.is_set():
                    data, _ = stream.read(512)
                    # data is shape (512, 1); extract the single channel as 1D
                    frame = data[:, 0]
                    t_now = time.time()
                    pipe.vad.push(frame, t_now, on_segment)
            except KeyboardInterrupt:
                print("\n[mic] stopping", flush=True)
            finally:
                pipe.vad.flush(on_segment)
                pipe.report()

    threading.Thread(target=mic_capture_loop, daemon=True).start()
    try:
        pipe.overlay.root.mainloop()
    except KeyboardInterrupt:
        pipe._stop.set()


if __name__ == "__main__":
    main()
