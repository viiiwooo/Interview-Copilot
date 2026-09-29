"""Smoke-test the running LLM server (LLM_BASE_URL from app/config.py).

Usage: .venv\\Scripts\\python.exe -m run.test_llm   (or python run/test_llm.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import LLM_BASE_URL, LLM_SYSTEM_PROMPT  # noqa: E402
from app.llm_client import ask, resolve_model  # noqa: E402

model = resolve_model(LLM_BASE_URL)
print("server:", LLM_BASE_URL, "model:", model)
r = ask(LLM_BASE_URL, model, LLM_SYSTEM_PROMPT, "Что такое TCP? Коротко.", max_tokens=120)
print(f"TTFT={r.ttft:.2f}s  total={r.total:.2f}s  tokens={r.n_tokens}  tps={r.tps:.1f}")
print("text:", r.text[:300])
