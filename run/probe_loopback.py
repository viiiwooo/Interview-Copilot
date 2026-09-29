"""Quick loopback capture sanity check: record 3s from the stereo mix device,
report RMS + whether it's mostly silence."""
import sys
import numpy as np
import sounddevice as sd

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SR = 16000
DUR = 3.0

print(f"Recording {DUR}s from device {DEV}: {sd.query_devices(DEV).get('name','')}")
data = sd.rec(int(SR * DUR), samplerate=SR, channels=1, dtype="int16", device=DEV)
sd.wait()
x = data[:, 0]
rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
peak = int(np.max(np.abs(x)))
print(f"RMS={rms:.1f}  peak={peak}  (silence ~0, speech ~500-5000)")
# save for inspection
import soundfile as sf
sf.write("run/loopback_probe.wav", x, SR, subtype="PCM_16")
print("saved run/loopback_probe.wav")
