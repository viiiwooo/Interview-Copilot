"""Download + verify GigaAM v3 e2e-rnnt checkpoint, then time a load."""
import os, sys, time, torch

t0 = time.time()
import gigaam
print(f"[bootstrap] import gigaam {time.time()-t0:.1f}s", flush=True)

name = "v3_e2e_rnnt"
t1 = time.time()
model = gigaam.load_model(name, fp16_encoder=True, device="cuda")
print(f"[bootstrap] load_model({name}) {time.time()-t1:.1f}s  (includes .ckpt + tokenizer download)", flush=True)
print(f"[bootstrap] model type={type(model).__name__} cfg.model_name={getattr(model.cfg,'model_name',None)}", flush=True)

# Save a tiny probe so the main pipeline can reuse the warm cache.
print("[bootstrap] OK", flush=True)
