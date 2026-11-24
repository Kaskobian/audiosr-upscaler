"""Compare AudioSR outputs (.npy, 48 kHz mono) against a reference.

Prints SNR and log-spectral distance (LSD, dB; lower = closer), overall and above 12 kHz,
where AudioSR generates content. Usage: python compare.py reference.npy other.npy ...
"""
import os
import sys

import numpy as np
from scipy import signal


def spectra(x):
    f, _, z = signal.stft(x, 48000, nperseg=2048)
    return f, 10 * np.log10(np.abs(z) ** 2 + 1e-10)


def lsd(a, b, mask):
    d = (a[mask] - b[mask]) ** 2
    return float(np.mean(np.sqrt(np.mean(d, axis=0))))


ref = np.load(sys.argv[1])
f, ref_db = spectra(ref)
for path in sys.argv[2:]:
    x = np.load(path)
    n = min(len(ref), len(x))
    _, x_db = spectra(x[:n])
    _, r_db = spectra(ref[:n])
    snr = 10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum((ref[:n] - x[:n]) ** 2), 1e-20))
    print(f"{os.path.basename(path):26s} SNR {snr:6.1f} dB | LSD all {lsd(r_db, x_db, f >= 0):5.2f} dB"
          f" | LSD >12k {lsd(r_db, x_db, f >= 12000):5.2f} dB")
