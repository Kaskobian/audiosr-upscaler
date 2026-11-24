"""Time and peak-VRAM per AudioSR stage, with optional speedups; saves the output for comparison.

Usage: python profile_stages.py [--fast] [--half sampling,decode,vocoder] [--save out.npy] file
"""
import argparse
import functools
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load

import audiosr.pipeline as pipeline  # noqa: E402
from audiosr import build_model, super_resolution  # noqa: E402
from audiosr.latent_diffusion.models.ddpm import LatentDiffusion  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speedups  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--fast", action="store_true", help="sliced attention + slim")
p.add_argument("--half", default="")
p.add_argument("--save", default="")
p.add_argument("--steps", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("file")
a = p.parse_args()


def timed(owner, name, label):
    fn = getattr(owner, name)

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        r = fn(*args, **kw)
        torch.cuda.synchronize()
        print(f"  {label:30s} {time.time() - t0:7.2f}s  peak {torch.cuda.max_memory_allocated() / 1024**3:6.2f} GB",
              flush=True)
        return r

    setattr(owner, name, wrapper)


torch.set_float32_matmul_precision("high")
model = build_model(model_name="basic", device="cuda")
if a.fast:
    print("  freed by slimming: %.2f GB" % speedups.apply(model, [s for s in a.half.split(",") if s]))
for owner, name, label in [(LatentDiffusion, "get_input", "prep + conditioning"),
                           (model, "sample_log", "sampling"),
                           (model, "decode_first_stage", "VAE decode"),
                           (model, "mel_spectrogram_to_waveform", "vocoder"),
                           (LatentDiffusion, "postprocessing", "postprocessing")]:
    timed(owner, name, label)

# warm-up on a short clip so first-call CUDA/cuDNN setup doesn't count
x, sr = sf.read(a.file, dtype="float32", always_2d=True)
import tempfile  # noqa: E402
warm = os.path.join(tempfile.gettempdir(), "audiosr_warmup.wav")
sf.write(warm, x[: sr * 3], sr)
print("  (warm-up)")
super_resolution(model, warm, seed=1, guidance_scale=3.5, ddim_steps=5, latent_t_per_second=12.8)
print("  (measured run)")
t0 = time.time()
w = super_resolution(model, a.file, seed=a.seed, guidance_scale=3.5, ddim_steps=a.steps, latent_t_per_second=12.8)
torch.cuda.synchronize()
print(f"  TOTAL {time.time() - t0:.1f}s")
if a.save:
    np.save(a.save, np.asarray(w[0, 0], dtype=np.float32))
