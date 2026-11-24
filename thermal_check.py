"""Alternate plain AudioSR and engine runs on one clip, logging GPU clock/temperature,
to separate engine cost from laptop thermal throttling. Usage: python thermal_check.py file.wav"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import speedups  # noqa: E402


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from audiosr import build_model, super_resolution  # noqa: E402


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return out


x, sr = sf.read(sys.argv[1], start=60 * 48000, frames=30 * 48000, dtype="float32", always_2d=True)
x = x.mean(1, keepdims=True)
path = os.path.join(tempfile.mkdtemp(), "clip.wav")
sf.write(path, x, sr, subtype="FLOAT")

torch.set_float32_matmul_precision("high")
model = build_model(model_name="basic", device="cuda")
speedups.apply(model, ("sampling", "decode", "vocoder"))


def run_model(p):
    return super_resolution(model, p, seed=42, guidance_scale=3.5, ddim_steps=50, latent_t_per_second=12.8)[0, 0]


run_model(path)
for name in ["plain", "engine", "plain", "engine"]:
    before = gpu_state()
    t0 = time.time()
    if name == "plain":
        run_model(path)
    else:
        engine.upscale(x, sr, run_model, stereo_mode="mono", chunk_seconds=None)
    torch.cuda.synchronize()
    print(f"{name:7s} {time.time() - t0:6.1f} s | GPU before [{before}] after [{gpu_state()}]", flush=True)
