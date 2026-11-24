"""Audio I/O and the music-aware pipeline around the AudioSR model.

The model (AudioSR) only ever sees one mono 48 kHz clip at a time. Everything here is
plain numpy/scipy so it can be tested without a GPU:

  read any format -> stereo handling (mono / mid-side / left-right) -> overlapping chunks
  -> model -> level match each chunk to the original -> keep the original audio up to the
  source's real bandwidth, use the model only above it -> crossfade chunks -> write any format
"""
import os
import tempfile
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy import signal

OUT_SR = 48000
INPUT_EXTS = (".wav", ".flac", ".mp3", ".m4a")
LOSSY_EXTS = (".mp3", ".m4a", ".aac", ".ogg", ".opus")
OUTPUT_FORMATS = {
    "WAV (24-bit)": ".wav",
    "FLAC (24-bit)": ".flac",
    "MP3 (320 kbps)": ".mp3",
    "M4A (AAC 256 kbps)": ".m4a",
}
STEREO_MODES = {
    "Stereo (mid/side, fast)": "ms",
    "Stereo full width (mid + side, 2x time)": "ms_full",
    "Stereo (left + right, 2x time)": "lr",
    "Mono": "mono",
}


class Cancelled(Exception):
    pass


# --- reading / writing -----------------------------------------------------------
def audio_info(path):
    """(duration_seconds, sample_rate, channels) without decoding the whole file."""
    try:
        info = sf.info(path)
        return info.duration, info.samplerate, info.channels
    except Exception:
        import av
        with av.open(path) as container:
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = (container.duration or 0) / 1_000_000
            return duration, stream.codec_context.sample_rate, stream.codec_context.channels


def read_audio(path):
    """Float32 array [frames, channels] and its sample rate."""
    try:
        return sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        # libsndfile can't decode AAC (M4A); PyAV ships its own FFmpeg for that.
        import av
        with av.open(path) as container:
            stream = container.streams.audio[0]
            sr = stream.codec_context.sample_rate
            resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout.name, rate=sr)
            parts = []
            for frame in container.decode(stream):
                parts.extend(f.to_ndarray() for f in resampler.resample(frame))
            parts.extend(f.to_ndarray() for f in resampler.resample(None))
        if not parts:
            raise ValueError("No audio found in " + os.path.basename(path))
        return np.concatenate(parts, axis=1).T.astype(np.float32), sr


def write_audio(path, audio, sr=OUT_SR):
    """Writes [frames, channels] float audio; format comes from the file extension.

    Restored highs can push a loud master over full scale. Rather than clip, the whole
    track is turned down just enough (-1 dB peak headroom for lossy codecs, which overshoot
    when decoded). Returns that gain change in dB (0 if none was needed).
    """
    ext = os.path.splitext(path)[1].lower()
    ceiling = 0.98 if ext in (".wav", ".flac") else 0.89
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    gain = ceiling / peak if peak > ceiling else 1.0
    audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
    if ext in (".wav", ".flac"):
        sf.write(path, audio, sr, subtype="PCM_24")
    elif ext == ".mp3":
        sf.write(path, audio, sr, format="MP3", compression_level=0.0, bitrate_mode="CONSTANT")
    elif ext == ".m4a":
        import av
        layout = "stereo" if audio.shape[1] == 2 else "mono"
        with av.open(path, "w", format="mp4") as container:
            stream = container.add_stream("aac", rate=sr, layout=layout)
            stream.bit_rate = 256_000
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio.T), format="fltp", layout=layout)
            frame.sample_rate = sr
            frame.pts = 0  # lets the muxer record the encoder delay so players trim it
            frame.time_base = Fraction(1, sr)
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    else:
        raise ValueError("Unsupported output format: " + ext)
    return float(20 * np.log10(gain))


# --- analysis ----------------------------------------------------------------------
def to_48k(audio, sr):
    if sr == OUT_SR:
        return audio.astype(np.float32)
    g = np.gcd(int(sr), OUT_SR)
    return signal.resample_poly(audio, OUT_SR // g, int(sr) // g, axis=0).astype(np.float32)


def detect_bandwidth(mono, sr):
    """Highest frequency (Hz) that holds real content. MP3/AAC encoders cut everything
    above ~16-20 kHz, which shows up as a cliff in the average spectrum."""
    nyquist = sr / 2
    if len(mono) < 4096:
        return nyquist
    f, _, z = signal.stft(mono, sr, nperseg=4096)
    power = np.mean(np.abs(z) ** 2, axis=1)
    db = 10 * np.log10(np.convolve(power, np.ones(5) / 5, mode="same") + 1e-20)
    # 1) Codec cutoff: a cliff of 25+ dB within 400 Hz somewhere above 5 kHz.
    step = max(1, int(round(400 / (f[1] - f[0]))))
    drops = db[:-step] - db[step:]
    drops[f[:-step] < 5000] = 0
    i = int(np.argmax(drops))
    if drops[i] > 25:
        # pin the cliff to the single steepest bin inside that 400 Hz window
        j = i + int(np.argmax(db[i:i + step] - db[i + 1:i + step + 1]))
        return float(f[j])
    # 2) Gentle roll-off (old or band-limited recordings): last frequency within 65 dB of the mids.
    mid = (f >= 500) & (f <= 4000)
    if not mid.any():
        return nyquist
    alive = np.nonzero(db > np.median(db[mid]) - 65)[0]
    if len(alive) == 0:
        return nyquist
    return float(min(f[alive[-1]], nyquist))


def lowpass(audio, cutoff, sr, order=8):
    if cutoff >= sr / 2 * 0.98:
        return audio
    sos = signal.butter(order, cutoff, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def match_level(generated, original, bandwidth):
    """AudioSR normalises every clip to 0.5 peak. Scale it back so its low band (which it
    copies from the input) lines up with the original (least squares)."""
    hi = min(max(bandwidth * 0.8, 1000), 6000)
    sos = signal.butter(4, [150, hi], btype="band", fs=OUT_SR, output="sos")
    gb = signal.sosfiltfilt(sos, generated)
    ob = signal.sosfiltfilt(sos, original)
    denom = float(np.dot(gb, gb))
    return generated * (float(np.dot(gb, ob)) / denom if denom > 1e-12 else 1.0)


def merge_bands(original, generated, bandwidth, transition=300):
    """Original audio below the source bandwidth, model output above it, with a smooth
    crossover ending just under the cutoff. Keeps every real detail of the input; the model
    only adds what's missing. Lossy sources use a wider transition (codec holes sit there)."""
    if bandwidth >= OUT_SR / 2 - 200:
        return original
    kw = dict(fs=OUT_SR, nperseg=2048, noverlap=1536)
    f, _, X = signal.stft(original, **kw)
    _, _, Y = signal.stft(generated, **kw)
    lo, hi = bandwidth - transition, bandwidth - 50
    t = np.clip((f - lo) / (hi - lo), 0, 1)
    keep = (0.5 + 0.5 * np.cos(np.pi * t))[:, None]  # 1 below lo -> 0 above hi
    _, out = signal.istft(keep * X + (1 - keep) * Y, **kw)
    out = out[: len(original)]
    if len(out) < len(original):
        out = np.pad(out, (0, len(original) - len(out)))
    return out.astype(np.float32)


def smoothed_spectrum_db(x, sr=OUT_SR):
    """Long-term power spectrum in dB, smoothed over 1/6 octave."""
    f, p = signal.welch(x, sr, nperseg=8192)
    lo = np.searchsorted(f, f / 2 ** (1 / 12))
    hi = np.maximum(np.searchsorted(f, f * 2 ** (1 / 12), side="right"), lo + 1)
    # Direct averages: a running-sum shortcut loses all precision at the quiet top end of music.
    smooth = np.array([p[a:b].mean() for a, b in zip(lo, hi)])
    return f, 10 * np.log10(smooth + 1e-30)


def continue_rolloff(track, bandwidth, max_boost_db=3.0, max_cut_db=40.0):
    """Shape the generated band (above `bandwidth`) so its long-term level continues the
    source's own high-frequency roll-off, like the envelope adjustment in AAC's spectral band
    replication. AudioSR tends to generate the top band far too loud for music (measured
    ~20 dB above the natural trend), which sounds like hiss. Static EQ, so the model's
    moment-to-moment detail and transients are kept."""
    if bandwidth >= OUT_SR / 2 - 200 or bandwidth < 4000:
        return track
    f, db = smoothed_spectrum_db(track)
    # Fit the real spectrum's trend; stop 1 kHz short of the cutoff so the 1/6-octave smoothing
    # can't pull the generated band's level into the fit.
    f_end = bandwidth - 1000
    fit = (f >= max(bandwidth - 5000, 2000)) & (f <= f_end)
    if fit.sum() < 8:
        return track
    slope, icept = np.polyfit(f[fit], db[fit], 1)
    slope = float(np.clip(slope, -0.006, 0.0))           # musical range: flat to -6 dB per kHz
    target = (icept + slope * f_end) + slope * (f - f_end)  # anchored at the fit's top
    corr = np.zeros_like(f)
    above = f >= bandwidth
    corr[above] = np.clip(target[above] - db[above], -max_cut_db, max_boost_db)
    corr = np.convolve(corr, np.ones(9) / 9, mode="same")  # ~50 Hz smoothing, no ripples
    ramp = (f > bandwidth - 300) & (f < bandwidth)          # ease in across the crossover
    corr[ramp] *= (f[ramp] - (bandwidth - 300)) / 300
    corr[f <= bandwidth - 300] = 0.0
    fir = signal.firwin2(4095, f / (OUT_SR / 2), 10 ** (corr / 20), fs=2)
    return signal.fftconvolve(track, fir, mode="same").astype(np.float32)


def plan_chunks(n, chunk, overlap, align=1):
    """Chunks of `chunk` samples covering [0, n), each starting `chunk - overlap` after the
    last, so neighbours share `overlap` samples. Starts stay on multiples of `align` (the
    model's latent frame) when chunk and overlap are multiples of it.

    The last piece is moved earlier so it is (nearly) full length: a short last piece gets
    padded with silence by AudioSR and generates poorly. It then shares more than `overlap`
    with the piece before it."""
    if chunk is None or n <= chunk:
        return [(0, n)]
    hop = chunk - overlap
    chunks = [(s, min(s + chunk, n)) for s in range(0, n - overlap, hop)]
    s_last, e_last = chunks[-1]
    if e_last - s_last < chunk and len(chunks) > 1:
        start = int(np.ceil((n - chunk) / align) * align)
        chunks[-1] = (min(start, s_last), n)
    return chunks


def crossfade_weights(length, fade_in, fade_out):
    w = np.ones(length, dtype=np.float32)
    if fade_in:
        w[:fade_in] = np.sin(np.linspace(0, np.pi / 2, fade_in)) ** 2
    if fade_out:
        w[-fade_out:] = np.cos(np.linspace(0, np.pi / 2, fade_out)) ** 2
    return w


# --- pipeline ----------------------------------------------------------------------
def upscale(audio, sr, run_model, *, stereo_mode="ms", lossy=False, chunk_seconds=None,
            overlap_seconds=2.0, on_progress=None, should_cancel=None, bandwidth=None, align=1,
            natural_highs=True):
    """audio [frames, channels] at sr -> ([frames48, channels], info).

    run_model(wav_path, carry_samples) must return the AudioSR output for that mono 48 kHz
    file; carry_samples is how much of its start repeats the previous piece (0 = new track).
    on_progress(run_index, run_count, label) is called before each model run.
    """
    if audio.shape[1] == 1 or stereo_mode == "mono":
        stereo_mode = "mono"
        audio = audio.mean(axis=1, keepdims=True)
    elif audio.shape[1] > 2:  # surround: fold down to stereo
        audio = np.stack([audio[:, 0::2].mean(1), audio[:, 1::2].mean(1)], axis=1)

    if bandwidth is None:
        bandwidth = detect_bandwidth(audio.mean(axis=1), sr)
    x48 = to_48k(audio, sr)
    if bandwidth >= OUT_SR / 2 - 200:  # already full-band: nothing for the model to add
        out = x48 if stereo_mode != "mono" or x48.shape[1] == 1 else x48.mean(axis=1, keepdims=True)
        return out, dict(bandwidth=bandwidth, chunks=0, runs=0, stereo_mode=stereo_mode)
    # README tip: lossy codecs leave holes just under their cutoff that confuse the model.
    # A clean low-pass at the detected bandwidth gives it the edge it was trained on.
    model_in = to_48k(lowpass(audio, bandwidth * 0.98, sr), sr) if lossy else x48

    if stereo_mode in ("ms", "ms_full"):
        mid, side = (x48[:, 0] + x48[:, 1]) / 2, (x48[:, 0] - x48[:, 1]) / 2
        # "ms" keeps the side's original content (fast; restored highs sit in the centre).
        # "ms_full" restores the side too, so cymbals and ambience regain width up top. A side
        # that is (nearly) silent - a mono file saved as stereo - is never sent to the model.
        side_alive = np.sqrt(np.mean(side ** 2)) > 1e-3 * max(np.sqrt(np.mean(mid ** 2)), 1e-12)
        tracks = [(mid, (model_in[:, 0] + model_in[:, 1]) / 2, True),
                  (side, (model_in[:, 0] - model_in[:, 1]) / 2, stereo_mode == "ms_full" and side_alive)]
    else:
        tracks = [(x48[:, c], model_in[:, c], True) for c in range(x48.shape[1])]

    n = len(x48)
    overlap = int(overlap_seconds * OUT_SR)
    chunk = int(chunk_seconds * OUT_SR) if chunk_seconds else None
    chunks = plan_chunks(n, chunk, overlap, align)
    runs = sum(len(chunks) for _, _, process in tracks if process)
    run = 0
    outputs = []
    with tempfile.TemporaryDirectory(prefix="audiosr_") as tmp:
        for t, (orig, feed, process) in enumerate(tracks):
            if not process:
                outputs.append(orig.astype(np.float32))
                continue
            out = np.zeros(n, dtype=np.float32)
            # One gain for the whole track (peak 0.5, what AudioSR expects) so every piece
            # reaches the model at the same level and pieces can continue each other.
            feed_gain = 0.5 / max(float(np.abs(feed).max()), 1e-8)
            for k, (s, e) in enumerate(chunks):
                if should_cancel and should_cancel():
                    raise Cancelled()
                if on_progress:
                    label = f"section {k + 1} of {len(chunks)}" if len(chunks) > 1 else ""
                    if runs > len(chunks):  # two channels go through the model
                        names = ("left", "right") if stereo_mode == "lr" else ("centre", "sides")
                        label = names[t] + (", " + label if label else "")
                    on_progress(run, runs, label)
                path = os.path.join(tmp, f"t{t}_c{k}.wav")
                shared = chunks[k - 1][1] - s if k > 0 else 0  # samples repeated from the previous piece
                sf.write(path, feed[s:e] * feed_gain, OUT_SR, subtype="FLOAT")
                gen = np.asarray(run_model(path, shared), dtype=np.float32)[: e - s]
                if len(gen) < e - s:
                    gen = np.pad(gen, (0, e - s - len(gen)))
                gen = match_level(gen, orig[s:e], bandwidth)
                merged = merge_bands(orig[s:e], gen, bandwidth, transition=600 if lossy else 300)
                # Crossfade over the last `overlap` samples of each shared region. Where a piece
                # shares more than that (the moved last piece), the previous piece covers it alone.
                w = crossfade_weights(e - s, 0, overlap if k < len(chunks) - 1 else 0)
                if shared:
                    w[: shared - overlap] = 0
                    w[shared - overlap: shared] *= crossfade_weights(overlap, overlap, 0)
                out[s:e] += merged * w
                run += 1
            outputs.append(continue_rolloff(out, bandwidth) if natural_highs else out)

    if stereo_mode in ("ms", "ms_full"):
        mid, side = outputs
        result = np.stack([mid + side, mid - side], axis=1)
    else:
        result = np.stack(outputs, axis=1)
    info = dict(bandwidth=bandwidth, chunks=len(chunks), runs=runs, stereo_mode=stereo_mode)
    return result, info


# --- self-check (no GPU needed): python engine.py --------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    sr = 44100
    t = np.arange(int(sr * 25)) / sr
    music = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(len(t))).astype(np.float32)

    def brickwall(x, fc):  # what an MP3 encoder does above its cutoff
        spec = np.fft.rfft(x)
        spec[np.fft.rfftfreq(len(x), 1 / sr) > fc] = 0
        return np.fft.irfft(spec, len(x)).astype(np.float32)

    left, right = brickwall(music, 16000), brickwall(np.roll(music, 50), 16000)
    audio = np.stack([left, right], axis=1)

    assert abs(detect_bandwidth(left, sr) - 16000) < 1500, detect_bandwidth(left, sr)

    def fake_model(path, carry=0):  # mimics AudioSR: same low band, new top band, peak-normalised
        x, _ = sf.read(path, dtype="float32")
        y = x + 0.02 * signal.sosfiltfilt(signal.butter(4, 17000, "high", fs=OUT_SR, output="sos"),
                                         rng.standard_normal(len(x) + 4800))[: len(x)]
        return 0.5 * np.concatenate([y, np.zeros(4800)]) / np.abs(y).max()

    out, info = upscale(audio, sr, fake_model, stereo_mode="lr", chunk_seconds=8, overlap_seconds=1.5)
    ref = to_48k(audio, sr)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    sos = signal.butter(4, 10000, "low", fs=OUT_SR, output="sos")
    err = signal.sosfiltfilt(sos, out - ref, axis=0)
    snr = 10 * np.log10(np.sum(signal.sosfiltfilt(sos, ref, axis=0) ** 2) / np.sum(err ** 2))
    assert snr > 30, f"low band drifted from the original: {snr:.1f} dB"
    hf = signal.sosfiltfilt(signal.butter(4, 17500, "high", fs=OUT_SR, output="sos"), out, axis=0)
    assert np.sqrt(np.mean(hf ** 2)) > 1e-3, "model's high band was dropped"
    jumps = np.abs(np.diff(out[:, 0]))
    assert jumps.max() < 6 * np.percentile(jumps, 99.9), "click at a chunk join"

    # A model that generates a far-too-loud top band gets shaped back onto the source's roll-off.
    def loud_model(path, carry=0):
        x, _ = sf.read(path, dtype="float32")
        hiss = signal.sosfiltfilt(signal.butter(4, 16500, "high", fs=OUT_SR, output="sos"),
                                  rng.standard_normal(len(x)))
        y = x + 0.5 * np.abs(x).max() * hiss / np.abs(hiss).max()
        return 0.5 * y / np.abs(y).max()

    tilted = signal.sosfiltfilt(signal.butter(1, 3000, "low", fs=sr, output="sos"), music)
    src = brickwall(tilted, 16000)[:, None]
    f, db_src = smoothed_spectrum_db(to_48k(src, sr)[:, 0])
    loud, _ = upscale(src, sr, loud_model, stereo_mode="mono", natural_highs=False)
    tamed, _ = upscale(src, sr, loud_model, stereo_mode="mono", natural_highs=True)
    _, db_loud = smoothed_spectrum_db(loud[:, 0])
    _, db_tamed = smoothed_spectrum_db(tamed[:, 0])
    below = (f > 13000) & (f < 15500)
    top = (f > 17000) & (f < 20000)
    jump = lambda d: float(np.mean(d[top]) - np.mean(d[below]))  # noqa: E731
    assert jump(db_loud) > 5 and jump(db_tamed) < 0, (jump(db_loud), jump(db_tamed))
    # below 14 kHz: outside both the shaped band (from ~15.7 kHz) and the 1/6-octave measuring window
    assert np.allclose(db_tamed[f < 14000], db_loud[f < 14000], atol=0.1), "shaping touched the real band"
    print(f"  roll-off shaping: top band {jump(db_loud):+.1f} dB -> {jump(db_tamed):+.1f} dB vs the band below")
    print(f"self-test OK: {info['chunks']} chunks, bandwidth {info['bandwidth']:.0f} Hz, "
          f"low-band match {snr:.1f} dB")


if __name__ == "__main__":
    _self_test()
