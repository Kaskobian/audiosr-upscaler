"""Benchmark AudioSR stages (prep / sampling / decode), peak VRAM, and output drift between variants.

Usage: python bench.py [--sdpa] [--precision fp32|fp16|bf16] [--steps N] [--save DIR] files...
"""
import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load

import audiosr.latent_diffusion.models.ddim as ddim  # noqa: E402
from audiosr import build_model, super_resolution  # noqa: E402

marks = {}


class Timer:
    def __init__(self, it, total=None, **_):
        self.it = it

    def __iter__(self):
        torch.cuda.synchronize()
        marks["s0"] = time.time()
        for x in self.it:
            yield x
        torch.cuda.synchronize()
        marks["s1"] = time.time()


ddim.tqdm = Timer

p = argparse.ArgumentParser()
p.add_argument("--sdpa", action="store_true")
p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
p.add_argument("--steps", type=int, default=50)
p.add_argument("--save", default="")
p.add_argument("files", nargs="+")
a = p.parse_args()

if a.sdpa:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import speedups
    speedups.apply()

torch.set_float32_matmul_precision("high")
model = build_model(model_name="basic", device="cuda")
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.precision)

for f in a.files:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        if dtype:
            with torch.autocast("cuda", dtype=dtype):
                w = super_resolution(model, f, seed=42, guidance_scale=3.5, ddim_steps=a.steps,
                                     latent_t_per_second=12.8)
        else:
            w = super_resolution(model, f, seed=42, guidance_scale=3.5, ddim_steps=a.steps,
                                 latent_t_per_second=12.8)
        torch.cuda.synchronize()
        t1 = time.time()
        print("%-28s total %6.1fs | prep %5.1f | sampling %6.1f | decode %6.1f | peak VRAM %5.2f GB" % (
            os.path.basename(f)[:28], t1 - t0, marks["s0"] - t0, marks["s1"] - marks["s0"],
            t1 - marks["s1"], torch.cuda.max_memory_allocated() / 1024 ** 3), flush=True)
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            np.save(os.path.join(a.save, os.path.splitext(os.path.basename(f))[0] + ".npy"),
                    np.asarray(w[0, 0], dtype=np.float32))
    except torch.OutOfMemoryError:
        print("%-28s OUT OF MEMORY (peak %.2f GB)" % (os.path.basename(f)[:28],
                                                      torch.cuda.max_memory_allocated() / 1024 ** 3), flush=True)
