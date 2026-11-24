"""Tune and compare model presets against the original AudioSR on a stereo ground truth.

Cuts a full-band stereo excerpt at a cutoff, restores it with each variant, and reports the
log-spectral distance (dB, lower = closer to the real original) above the cutoff for the left
and right channels and for the side (stereo) signal.
Usage: python eval_models.py reference.wav start_s length_s cutoff_hz [--model basic|speech]
       [--variants name,name,...]
"""
import os
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
from sr_runner import AudioSRRunner, LATENT_HOP, OVERLAP_SECONDS, PIECE_SECONDS  # noqa: E402


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from audiosr import build_model, super_resolution  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a.lstrip("-").split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ref, start, length, cutoff = args[0], float(args[1]), float(args[2]), float(args[3])
model_name = opts.get("model", "basic")

info = sf.info(ref)
x, sr = sf.read(ref, start=int(start * info.samplerate), frames=int(length * info.samplerate),
                dtype="float32", always_2d=True)
if x.shape[1] == 1:
    x = np.repeat(x, 2, axis=1)
truth = engine.to_48k(x, sr)
spec = np.fft.rfft(x, axis=0)
spec[np.fft.rfftfreq(len(x), 1 / sr) > cutoff] = 0
degraded = np.fft.irfft(spec, len(x), axis=0).astype(np.float32)
true_bw = engine.detect_bandwidth(x.mean(1), sr)
top = min(true_bw, 23500)


def lsd(a, b):
    f, _, za = engine.signal.stft(a, 48000, nperseg=2048)
    _, _, zb = engine.signal.stft(np.asarray(b, dtype=np.float64), 48000, nperseg=2048)
    m = (f > cutoff) & (f <= top)
    da, db = 10 * np.log10(np.abs(za[m]) ** 2 + 1e-10), 10 * np.log10(np.abs(zb[m]) ** 2 + 1e-10)
    return float(np.mean(np.sqrt(np.mean((da - db) ** 2, axis=0))))


def report(name, out, seconds):
    n = min(len(out), len(truth))
    o, t = out[:n], truth[:n]
    left, right = lsd(t[:, 0], o[:, 0]), lsd(t[:, 1], o[:, 1])
    side = lsd((t[:, 0] - t[:, 1]) / 2, (o[:, 0] - o[:, 1]) / 2)
    print(f"{name:44s} L {left:5.2f} | R {right:5.2f} | avg {(left + right) / 2:5.2f} | side {side:5.2f} dB"
          f" | {seconds:5.0f} s", flush=True)


print(f"{os.path.basename(ref)} {start:.0f}-{start + length:.0f} s, cut at {cutoff / 1000:.1f} kHz, "
      f"real content to {true_bw / 1000:.1f} kHz, model '{model_name}'", flush=True)
model = build_model(model_name=model_name, device="cuda")
speedups.apply(model, ("sampling", "decode", "vocoder"))
report("degraded input (no processing)", engine.to_48k(degraded, sr), 0)

tmp = tempfile.mkdtemp()
VARIANTS = {
    # name: (stereo mode, guidance, steps)
    "original": None,
    "ms_g3.5": ("ms", 3.5, 50),
    "msfull_g3.5": ("ms_full", 3.5, 50),
    "msfull_g2.5": ("ms_full", 2.5, 50),
    "msfull_g5": ("ms_full", 5.0, 50),
    "mono_g3.5": ("mono", 3.5, 50),
    "mono_g2.5": ("mono", 2.5, 50),
    "mono_g5": ("mono", 5.0, 50),
}
wanted = opts.get("variants", "original,ms_g3.5,msfull_g3.5,msfull_g2.5,msfull_g5").split(",")
for name in wanted:
    t0 = time.time()
    if VARIANTS[name] is None:
        # the year-old way: one mono pass of plain AudioSR, same file for both channels
        p = os.path.join(tmp, "deg.wav")
        sf.write(p, degraded.mean(1), sr, subtype="FLOAT")
        w = np.asarray(super_resolution(model, p, seed=42, guidance_scale=3.5, ddim_steps=50,
                                        latent_t_per_second=12.8)[0, 0], dtype=np.float32)[: len(truth)]
        w = engine.match_level(w, truth.mean(1)[: len(w)], cutoff)
        report("original AudioSR (one mono pass)", np.stack([w, w], 1), time.time() - t0)
        continue
    mode, guidance, steps = VARIANTS[name]
    runner = AudioSRRunner(model, steps=steps, guidance=guidance, seed=42, continuity=True)
    out, meta = engine.upscale(degraded, sr, runner, stereo_mode=mode, chunk_seconds=PIECE_SECONDS,
                               overlap_seconds=OVERLAP_SECONDS, align=LATENT_HOP)
    if out.shape[1] == 1:
        out = np.repeat(out, 2, axis=1)
    report(f"app: {mode}, guidance {guidance}, {meta['runs']} runs", out, time.time() - t0)
