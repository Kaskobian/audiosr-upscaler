"""Objective quality test: cut a full-band track at a cutoff, restore it, compare to the original.

LSD = log-spectral distance in dB vs the true original (lower is better), reported for the
restored band (above the cutoff) and the kept band (below it).
Usage: python eval.py reference.wav [start_s] [length_s] [cutoff_hz]
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


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from audiosr import build_model, super_resolution  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ref_path = args[0]
start = float(args[1]) if len(args) > 1 else 60
length = float(args[2]) if len(args) > 2 else 30
cutoff = float(args[3]) if len(args) > 3 else 11000

info = sf.info(ref_path)
x, sr = sf.read(ref_path, start=int(start * info.samplerate), frames=int(length * info.samplerate),
                dtype="float32", always_2d=True)
truth = engine.to_48k(x.mean(axis=1, keepdims=True), sr)[:, 0]
print(f"reference: {os.path.basename(ref_path)} {sr} Hz, real bandwidth "
      f"{engine.detect_bandwidth(x.mean(1), sr) / 1000:.1f} kHz; testing {length:.0f} s cut at {cutoff / 1000:.0f} kHz")

spec = np.fft.rfft(x, axis=0)
spec[np.fft.rfftfreq(len(x), 1 / sr) > cutoff] = 0
degraded = np.fft.irfft(spec, len(x), axis=0).astype(np.float32).mean(axis=1, keepdims=True)
tmp = tempfile.mkdtemp(prefix="audiosr_eval_")
deg_path = os.path.join(tmp, "degraded.wav")
sf.write(deg_path, degraded, sr, subtype="FLOAT")

torch.set_float32_matmul_precision("high")
model = build_model(model_name="basic", device="cuda")
speedups.apply(model, ("sampling", "decode", "vocoder"))


def run_model(path, carry=0):
    return super_resolution(model, path, seed=42, guidance_scale=3.5, ddim_steps=50, latent_t_per_second=12.8)[0, 0]


run_model(deg_path)  # warm-up so timings are comparable


def lsd(est):
    n = min(len(est), len(truth))
    f, _, a = engine.signal.stft(truth[:n], 48000, nperseg=2048)
    _, _, b = engine.signal.stft(np.asarray(est[:n], dtype=np.float64), 48000, nperseg=2048)
    da, db = 10 * np.log10(np.abs(a) ** 2 + 1e-10), 10 * np.log10(np.abs(b) ** 2 + 1e-10)

    def band(mask):
        return float(np.mean(np.sqrt(np.mean((da[mask] - db[mask]) ** 2, axis=0))))

    return band(f > cutoff), band(f <= cutoff)


def report(name, est, seconds):
    hi, lo = lsd(est)
    print(f"{name:34s} LSD restored band {hi:5.2f} dB | kept band {lo:5.2f} dB | {seconds:6.1f} s", flush=True)


report("degraded input (no processing)", engine.to_48k(degraded, sr)[:, 0], 0)

t0 = time.time()
raw = run_model(deg_path)
raw = engine.match_level(np.asarray(raw[: len(truth)], dtype=np.float32), truth, cutoff)  # level only, for fairness
report("original AudioSR (one piece)", raw, time.time() - t0)

if "--rolloff" in sys.argv:
    from sr_runner import AudioSRRunner, LATENT_HOP, OVERLAP_SECONDS, PIECE_SECONDS
    for natural in (False, True):
        runner = AudioSRRunner(model, seed=42, continuity=True)
        t0 = time.time()
        out, meta = engine.upscale(degraded, sr, runner, stereo_mode="mono", chunk_seconds=PIECE_SECONDS,
                                   overlap_seconds=OVERLAP_SECONDS, align=LATENT_HOP, natural_highs=natural)
        report(f"app pipeline, roll-off shaping {'ON' if natural else 'off'}", out[:, 0], time.time() - t0)
    sys.exit()
variants = [(None, 2.0), (10.24, 2.0)] if "--quick" in sys.argv else [(None, 2.0), (10.24, 2.0), (5.12, 1.0)]
for chunk, overlap in variants:
    t0 = time.time()
    out, meta = engine.upscale(degraded, sr, run_model, stereo_mode="mono", lossy=False,
                               chunk_seconds=chunk, overlap_seconds=overlap)
    label = "new engine, one piece" if chunk is None else f"new engine, {chunk:g} s pieces"
    report(label + f" ({meta['runs']} runs)", out[:, 0], time.time() - t0)
