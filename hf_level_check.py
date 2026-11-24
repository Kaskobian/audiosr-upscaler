"""Is the loud generated top band from AudioSR itself or from our pipeline?
Runs a 20 s excerpt through plain AudioSR and through the engine (shared / per-piece gain)
and prints band levels relative to 1-4 kHz. Usage: python hf_level_check.py song.mp3 [start_s]"""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import speedups  # noqa: E402
from sr_runner import AudioSRRunner, LATENT_HOP, OVERLAP_SECONDS, PIECE_SECONDS  # noqa: E402


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from audiosr import build_model, super_resolution  # noqa: E402

BANDS = [(8000, 12000), (12000, 15000), (15000, 16500), (16800, 18000), (18000, 20000), (20000, 22000),
         (22000, 24000)]


def bands(mono48):
    f, p = signal.welch(mono48, 48000, nperseg=8192)
    db = lambda lo, hi: 10 * np.log10(np.mean(p[(f >= lo) & (f < hi)]) + 1e-20)  # noqa: E731
    ref = db(1000, 4000)
    return " ".join(f"{db(lo, hi) - ref:6.1f}" for lo, hi in BANDS)


audio, sr = engine.read_audio(sys.argv[1])
start = float(sys.argv[2]) if len(sys.argv) > 2 else 60
audio = audio[int(start * sr): int((start + 20) * sr)]
mono = audio.mean(axis=1, keepdims=True)
bw = engine.detect_bandwidth(mono[:, 0], sr)
tmp = tempfile.mkdtemp()
path = os.path.join(tmp, "excerpt.wav")
sf.write(path, mono, sr, subtype="FLOAT")

model = build_model(model_name="basic", device="cuda")
speedups.apply(model, ("sampling", "decode", "vocoder"))
print(f"bandwidth {bw / 1000:.1f} kHz")
print("band (dB vs 1-4k):            8-12k 12-15k 15-16.5 16.8-18 18-20k 20-22k 22-24k")
print("input                       ", bands(engine.to_48k(mono, sr)[:, 0]), flush=True)
raw = super_resolution(model, path, seed=42, guidance_scale=3.5, ddim_steps=50, latent_t_per_second=12.8)[0, 0]
print("plain AudioSR (one piece)   ", bands(np.asarray(raw)), flush=True)
for name, shared in [("engine, shared gain", True), ("engine, per-piece gain", False)]:
    runner = AudioSRRunner(model, seed=42, continuity=True, shared_gain=shared)
    out, _ = engine.upscale(mono, sr, runner, stereo_mode="mono", lossy=True, chunk_seconds=PIECE_SECONDS,
                            overlap_seconds=OVERLAP_SECONDS, bandwidth=bw, align=LATENT_HOP)
    print(f"{name:28s}", bands(out[:, 0]), flush=True)
