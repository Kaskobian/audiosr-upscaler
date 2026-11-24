<p align="center"><img src="assets/hamster_256.png" width="128" alt="AudioSR Upscaler hamster"></p>

# AudioSR Upscaler

A Windows desktop app that restores the missing high frequencies of compressed or low-quality
audio (MP3s, phone recordings, old rips) and outputs 48 kHz, with optional **reference
remastering**. It builds on the [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution)
diffusion model and adds a processing pipeline that is faster, handles whole songs seamlessly,
keeps your original audio intact, and measurably beats the original AudioSR on music and speech.

> Independent project, not affiliated with the AudioSR authors. AudioSR is MIT-licensed; see
> [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Download

Get **`AudioSR-Upscaler-Setup-<version>.exe`** from [Releases](../../releases), run it, and click
**Install**. It installs to `C:\Program Files\AudioSR` and adds Start menu and desktop shortcuts;
uninstall from **Settings → Apps**. The setup is not code-signed yet, so Windows SmartScreen
may warn on first run (**More info → Run anyway**).

**Requirements:** Windows 10/11 64-bit. An NVIDIA GPU is strongly recommended (tested on an
RTX 5070 Ti Laptop, 12 GB); CPU works but is much slower. The first time you use each model,
it downloads that model's weights from Hugging Face (5.75 GB each).

## Features

- **Three models:** **General**, **Music** (full-width stereo restoration) and **Speech / Voice**
  (AudioSR's speech checkpoint), each with settings tuned on measurements.
- **Whole songs, seamless:** long files are processed in overlapping 10 s sections; each section
  *continues* the previous one's generated highs (diffusion inpainting), so there are no joins.
- **Keeps your audio:** detects where the source was actually cut off and keeps everything below
  that untouched; the AI only fills in what is missing, with a natural high-frequency roll-off.
- **Reference remaster:** match a reference track's tonal balance, stereo width and loudness
  (LUFS), with a strength slider and a safety limiter.
- **Formats:** WAV, FLAC, MP3, M4A in and out (24-bit WAV/FLAC, 320 kbps MP3, 256 kbps AAC).
- **Faster and lighter:** bf16 tensor cores, unused model parts kept in system RAM, sliced
  attention; long files never run out of VRAM.
- Shows the detected hardware, real progress with time remaining, and a batch mode
  (`AudioSR_Upscaler.exe --batch in.wav out.flac --model Music`).

## Measured against the original AudioSR

Log-spectral distance to the true original in the restored band (dB, lower is better),
on material deliberately cut off and then restored:

| Test | Original AudioSR | This app |
|---|---|---|
| Music, 20 s stereo, cut at 11 kHz | 12.65 (no stereo highs restored: side 28.9) | **11.19** (side 10.1) |
| Speech, 29 s voice, cut at 4 kHz | 9.16 (worse than the untouched input, 7.82) | **4.44** |
| Real audio below the cutoff (music) | changed by 2.57 dB | kept within 1.0 dB |
| Disagreement at section joins | 7.5 dB (independent sections) | **3.1 dB** |
| 53 s song on an RTX 5070 Ti Laptop | 497 s, 13.6 GB VRAM | **168 s, 9.5 GB** |

The scripts that produced these numbers are in the repo (`eval_models.py`, `eval.py`,
`seam_eval.py`, `seam_audit.py`, `profile_stages.py`). The speech test used synthetic
(text-to-speech) voice.

## Build from source

```powershell
python -m venv .venv                                  # Python 3.11
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python audiosr_gui.py                   # run the app
powershell -File build_app.ps1 -Python .venv\Scripts\python.exe   # dist\AudioSR_Upscaler
powershell -File installer\build.ps1 -Version 1.0.0               # single setup .exe
```

The installer build needs the [WiX Toolset](https://wixtoolset.org) 5
(`dotnet tool install --global wix --version 5.0.2` and
`wix extension add -g WixToolset.Bal.wixext/5.0.2`).

| File | What it does |
|---|---|
| `audiosr_gui.py` | The app window and batch mode |
| `engine.py` | Audio I/O, sections, band merging, roll-off, stereo handling |
| `sr_runner.py` | Runs AudioSR section by section with continuation |
| `speedups.py` | bf16, VRAM offload, sliced attention |
| `remaster.py` | Reference remastering |
| `installer/` | WiX installer and setup bundle |

## License

[MIT](LICENSE). AudioSR and the other bundled components keep their own licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
