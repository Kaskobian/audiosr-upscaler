"""Audit an upscaled file for audible seams and timing problems.

  brightness jump at joins: change in the restored band's level relative to the real band
      just below the cutoff, across each section join, vs the same measure at other moments.
      A join that stands out (high percentile) is an audible "step" in brightness.
  HF timing: lag (ms) between the restored band's envelope and the real band's envelope.
      Non-zero means the generated highs are early/late -> smeared, stuttery transients.
Usage: python seam_audit.py source_file processed_file
"""
import os
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
from sr_runner import LATENT_HOP, OVERLAP_SECONDS, PIECE_SECONDS  # noqa: E402

SR = 48000
src, ssr = engine.read_audio(sys.argv[1])
out, osr = engine.read_audio(sys.argv[2])
bw = engine.detect_bandwidth(src.mean(1), ssr)
mono = out.mean(axis=1)
n = len(mono)


def band_env(x, lo, hi, frame=0.02):
    sos = signal.butter(6, [lo, min(hi, SR / 2 - 100)], "band", fs=SR, output="sos")
    y = signal.sosfiltfilt(sos, x)
    hop = int(frame * SR)
    e = np.sqrt(np.mean(y[: len(y) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    return e, hop


hi_env, hop = band_env(mono, bw + 200, 23500)
lo_env, _ = band_env(mono, max(bw - 3000, 2000), bw - 300)

# --- timing: cross-correlate onset strength of the two bands ---
d_hi = np.maximum(np.diff(np.log(hi_env + 1e-9)), 0)
d_lo = np.maximum(np.diff(np.log(lo_env + 1e-9)), 0)
d_hi, d_lo = d_hi - d_hi.mean(), d_lo - d_lo.mean()
fine_hi, fine_hop = band_env(mono, bw + 200, 23500, frame=0.002)
fine_lo, _ = band_env(mono, max(bw - 3000, 2000), bw - 300, frame=0.002)
a = np.maximum(np.diff(np.log(fine_hi + 1e-9)), 0)
b = np.maximum(np.diff(np.log(fine_lo + 1e-9)), 0)
a, b = a - a.mean(), b - b.mean()
max_lag = 50  # x 2 ms
xc = [np.dot(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) - max(0, k)]) for k in range(-max_lag, max_lag + 1)]
lag_ms = (int(np.argmax(xc)) - max_lag) * 2

# --- brightness steps at joins ---
ratio = 20 * np.log10((hi_env + 1e-9) / (lo_env + 1e-9))
ratio = np.convolve(ratio, np.ones(5) / 5, mode="same")  # 100 ms smoothing
span = int(0.6 / 0.02)  # compare 0.6 s before vs after
step = np.full(len(ratio), np.nan)
for i in range(span, len(ratio) - span):
    step[i] = abs(np.mean(ratio[i:i + span]) - np.mean(ratio[i - span:i]))
plan = engine.plan_chunks(n, int(PIECE_SECONDS * SR), int(OVERLAP_SECONDS * SR), LATENT_HOP)
ov = int(OVERLAP_SECONDS * SR)
joins = [(plan[k - 1][1] - ov // 2) // hop for k in range(1, len(plan))]
valid = step[~np.isnan(step)]
at_joins = np.array([step[j] for j in joins if span <= j < len(step) - span])
pct = [float(np.mean(valid < v) * 100) for v in at_joins]
# --- dips: restored band sagging in the middle of a crossfade (partly different content
# from the two sections adds with less than full power) ---
w = int(1.0 / 0.02)
dip = np.full(len(ratio), np.nan)
for i in range(w, len(ratio) - w):
    dip[i] = np.mean(np.r_[ratio[i - w:i - w // 2], ratio[i + w // 2:i + w]]) - ratio[i]
dvalid = dip[~np.isnan(dip)]
dips = np.array([dip[j] for j in joins if w <= j < len(dip) - w])
print(f"dip at join centres: mean {np.mean(dips):+.2f} dB (song-wide mean {np.mean(dvalid):+.2f} dB); "
      f"joins in the song's deepest 10%: {int(np.sum(dips > np.percentile(dvalid, 90)))} of {len(dips)}")
print(f"bandwidth {bw / 1000:.1f} kHz | {len(plan)} sections, {len(joins)} joins")
print(f"HF timing vs real band: {lag_ms:+d} ms (0 = aligned)")
print(f"brightness step at joins: median {np.median(at_joins):.2f} dB (song-wide median {np.median(valid):.2f} dB)")
print(f"joins that stand out (above the song's 90th percentile): {sum(p > 90 for p in pct)} of {len(pct)}; "
      f"worst at {max(pct):.0f}th percentile")
