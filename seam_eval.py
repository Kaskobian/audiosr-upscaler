"""Seam test: do neighbouring pieces agree on the generated highs where they overlap?

Cuts a full-band excerpt at a cutoff, restores it in 10.24 s pieces with each runner variant,
then reports:
  overlap disagreement: LSD (dB) between piece k-1's tail and piece k's head above the cutoff.
                        Independent pieces disagree a lot; continuation should drive it down.
  restored-band LSD:    final output vs the true original above the cutoff (quality).
Usage: python seam_eval.py reference.wav [start_s] [length_s] [cutoff_hz]
"""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import speedups  # noqa: E402
from sr_runner import AudioSRRunner, OVERLAP_SECONDS, PIECE_SECONDS  # noqa: E402


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from audiosr import build_model  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ref = args[0]
start = float(args[1]) if len(args) > 1 else 60
length = float(args[2]) if len(args) > 2 else 30
cutoff = float(args[3]) if len(args) > 3 else 11000

x, sr = sf.read(ref, start=int(start * 48000), frames=int(length * 48000), dtype="float32", always_2d=True)
x = x.mean(axis=1, keepdims=True)
truth = engine.to_48k(x, sr)[:, 0]
spec = np.fft.rfft(x, axis=0)
spec[np.fft.rfftfreq(len(x), 1 / sr) > cutoff] = 0
degraded = np.fft.irfft(spec, len(x), axis=0).astype(np.float32)
deg48 = engine.to_48k(degraded, sr)[:, 0]
bandwidth = engine.detect_bandwidth(degraded[:, 0], sr)

model = build_model(model_name="basic", device="cuda")
speedups.apply(model, ("sampling", "decode", "vocoder"))
chunk, overlap = int(PIECE_SECONDS * 48000), int(OVERLAP_SECONDS * 48000)
plan = engine.plan_chunks(len(deg48), chunk, overlap, 3840)
f = None


def hf_db(sig):
    global f
    f, _, z = engine.signal.stft(sig, 48000, nperseg=2048)
    return 10 * np.log10(np.abs(z[f > bandwidth + 300]) ** 2 + 1e-10)


def lsd(a, b):
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=0))))


variants = [("independent pieces (old)", False, False, 0),
            ("continuation, shared gain", True, True, 0),
            ("continuation, per-piece gain", True, False, 0)]
if "--pair" in sys.argv:
    variants = [("independent pieces (old)", False, True, 0),
                ("continuation", True, True, 0)]
elif "--soft" in sys.argv:
    variants = [("independent pieces (old)", False, True, 0),
                ("continuation, hard edge", True, True, 0),
                ("continuation, soft edge 8", True, True, 8),
                ("continuation, soft edge 16", True, True, 16)]
seeds = [int(s) for s in os.environ.get("SEAM_SEEDS", "42").split(",")]
for (name, cont, shared, edge), seed in [(v, s) for s in seeds for v in variants]:
    name = f"{name} [seed {seed}]"
    runner = AudioSRRunner(model, steps=50, guidance=3.5, seed=seed, continuity=cont, shared_gain=shared,
                           edge_frames=edge)
    pieces = []

    def run_model(path, carry):
        out = runner(path, carry)
        pieces.append(np.asarray(out, dtype=np.float32))
        return out

    t0 = time.time()
    out, _ = engine.upscale(degraded, sr, run_model, stereo_mode="mono", chunk_seconds=PIECE_SECONDS,
                            overlap_seconds=OVERLAP_SECONDS, bandwidth=bandwidth, align=3840)
    seconds = time.time() - t0
    seams = []
    for k in range(1, len(plan)):
        (s0, e0), (s1, e1) = plan[k - 1], plan[k]
        shared = e0 - s1  # the crossfade covers the last `overlap` samples of the shared region
        a = engine.match_level(pieces[k - 1][: e0 - s0], deg48[s0:e0], bandwidth)[-overlap:]
        b = engine.match_level(pieces[k][: e1 - s1], deg48[s1:e1], bandwidth)[shared - overlap:shared]
        seams.append(lsd(hf_db(a), hf_db(b)))
    quality = lsd(hf_db(truth), hf_db(out[:, 0]))
    print(f"{name:30s} overlap disagreement {np.mean(seams):5.2f} dB (per seam: "
          f"{', '.join(f'{s:.1f}' for s in seams)}) | restored-band LSD {quality:5.2f} dB | {seconds:5.0f} s",
          flush=True)
