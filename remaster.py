"""Reference remastering: make a track match a reference's tonal balance, stereo width and
loudness, then catch the peaks with a limiter. Everything is measured from the two files.

  1. Tonal balance: smoothed (1/6 octave) average spectra of both mids; the difference becomes a
     linear-phase EQ (broadband level removed, capped at +-12 dB, held flat above the band
     both files actually contain).
  2. Stereo width: side-to-mid energy ratio; only the side above 200 Hz is scaled (keeps bass
     centred), limited to 0.5x-2x.
  3. Loudness: integrated loudness (LUFS, ITU-R BS.1770) moved toward the reference's.
  4. Peaks: look-ahead limiter to -1 dBFS so the louder result doesn't clip.
`strength` (0-1) scales steps 1-3.
"""
import numpy as np
import pyloudnorm
from scipy import ndimage, signal

SR = 48000
MAX_EQ_DB = 12.0
CEILING = 0.89  # -1 dBFS


from engine import smoothed_spectrum_db  # noqa: E402  (1/6-octave, numerically safe)


def eq_curve(target, reference, limit_hz, strength):
    f, t = smoothed_spectrum_db(target)
    _, r = smoothed_spectrum_db(reference)
    diff = r - t
    mid = (f >= 200) & (f <= 5000)
    diff -= np.mean(diff[mid])                       # level is matched separately, by loudness
    usable = (f >= 25) & (f <= limit_hz)
    if not usable.any():
        return f, np.zeros_like(f)
    first, last = np.nonzero(usable)[0][[0, -1]]
    diff[:first], diff[last:] = diff[first], diff[last]  # hold flat outside the shared band
    return f, np.clip(diff * strength, -MAX_EQ_DB, MAX_EQ_DB)


def apply_eq(audio, f, gain_db, taps=4095):
    fir = signal.firwin2(taps, f / (SR / 2), 10 ** (gain_db / 20), fs=2)  # f runs 0..Nyquist
    return signal.fftconvolve(audio, fir[:, None], mode="same", axes=0).astype(np.float32)


def side_mid_ratio(audio):
    mid, side = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
    return float(np.sqrt(np.mean(side ** 2) / max(np.mean(mid ** 2), 1e-20)))


def set_width(audio, factor):
    mid, side = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
    sos = signal.butter(4, 200, "low", fs=SR, output="sos")
    low = signal.sosfiltfilt(sos, side)
    side = low + (side - low) * factor
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def limit(audio, ceiling=CEILING, window_ms=20):
    """Look-ahead limiter: gain falls smoothly before each peak and recovers after it.
    A centred minimum filter followed by an equal-width average can never exceed the needed
    reduction at a peak, so no sample ends up above the ceiling."""
    peak = np.max(np.abs(audio), axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))
    if need.min() >= 1.0:
        return audio, 0.0
    w = max(3, int(SR * window_ms / 1000) | 1)
    gain = ndimage.uniform_filter1d(ndimage.minimum_filter1d(need, w), w)
    gain = np.minimum(gain, need)  # guard against float rounding at the edges
    return (audio * gain[:, None]).astype(np.float32), float(-20 * np.log10(gain.min()))


def loudness(audio):
    meter = pyloudnorm.Meter(SR)
    try:
        return float(meter.integrated_loudness(audio if audio.shape[1] > 1 else audio[:, 0]))
    except ValueError:  # shorter than one 400 ms block
        return float("-inf")


def match(target, reference, *, strength=1.0, match_loudness=True, limit_hz=SR / 2):
    """target, reference: float arrays [frames, channels] at 48 kHz. Returns (audio, report)."""
    strength = float(np.clip(strength, 0, 1))
    mono = lambda a: a.mean(axis=1)  # noqa: E731
    report = {}

    f, gains = eq_curve(mono(target), mono(reference), limit_hz, strength)
    out = apply_eq(target, f, gains)
    bands = {"bass (<120 Hz)": f < 120, "mids (300 Hz-3 kHz)": (f >= 300) & (f <= 3000),
             "highs (>6 kHz)": f > 6000}
    report["eq"] = {name: float(np.mean(gains[m])) for name, m in bands.items() if m.any()}

    if out.shape[1] == 2 and reference.shape[1] == 2:
        before, ref_w = side_mid_ratio(out), side_mid_ratio(reference)
        factor = float(np.clip((ref_w / max(before, 1e-6)) ** strength, 0.5, 2.0))
        out = set_width(out, factor)
        report["width"] = factor

    lufs_before, lufs_ref = loudness(out), loudness(reference)
    report["lufs_before"], report["lufs_reference"] = lufs_before, lufs_ref
    if match_loudness and np.isfinite(lufs_before) and np.isfinite(lufs_ref):
        goal = lufs_before + (lufs_ref - lufs_before) * strength
        out = out * 10 ** ((goal - lufs_before) / 20)
    out, reduction = limit(out)
    report["limiter_db"] = reduction
    report["lufs_after"] = loudness(out)
    return out.astype(np.float32), report


def _self_test():
    rng = np.random.default_rng(1)
    n = SR * 20

    def pinkish(cut):
        x = signal.lfilter([1], [1, -0.97], rng.standard_normal(n))
        return signal.sosfiltfilt(signal.butter(4, cut, "low", fs=SR, output="sos"), x)

    common = pinkish(16000)
    reference = np.stack([common + 0.5 * pinkish(16000), common - 0.5 * pinkish(16000)], 1)
    reference = (0.5 * reference / np.abs(reference).max()).astype(np.float32)
    dull = signal.sosfiltfilt(signal.butter(2, 3000, "low", fs=SR, output="sos"), pinkish(16000))
    target = np.stack([dull + 0.1 * pinkish(16000), dull - 0.1 * pinkish(16000)], 1)
    target = (0.05 * target / np.abs(target).max()).astype(np.float32)

    out, rep = match(target, reference)
    f, t0 = smoothed_spectrum_db(target.mean(1))
    _, r = smoothed_spectrum_db(reference.mean(1))
    _, o = smoothed_spectrum_db(out.mean(1))
    band = (f > 100) & (f < 12000)
    err = lambda a: float(np.std((r - a)[band]))  # noqa: E731  (shape error, level removed)
    assert err(o) < err(t0) / 3, (err(t0), err(o))
    assert abs(rep["lufs_after"] - rep["lufs_reference"]) < 1.0 or rep["limiter_db"] > 0, rep
    assert np.abs(out).max() <= CEILING + 1e-4
    w_t, w_r, w_o = side_mid_ratio(target), side_mid_ratio(reference), side_mid_ratio(out)
    assert abs(w_o - w_r) < abs(w_t - w_r) / 2, (w_t, w_r, w_o)
    print(f"self-test OK: spectral shape error {err(t0):.1f} -> {err(o):.1f} dB | width "
          f"{w_t:.2f} -> {w_o:.2f} (ref {w_r:.2f}) | {rep['lufs_before']:.1f} -> {rep['lufs_after']:.1f} LUFS "
          f"(ref {rep['lufs_reference']:.1f}) | limiter {rep['limiter_db']:.1f} dB | peak {np.abs(out).max():.3f}")


if __name__ == "__main__":
    _self_test()
