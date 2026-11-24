"""Which torch\\lib DLLs does real processing actually load? Runs short jobs in every app mode
(GPU bf16, GPU fp32, CPU, both checkpoints) and lists DLLs from torch\\lib that were never
loaded - candidates to leave out of the installer. Usage: python dll_usage.py clip.wav"""
import ctypes
import ctypes.wintypes as wt
import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine  # noqa: E402
import speedups  # noqa: E402
from sr_runner import AudioSRRunner, LATENT_HOP, OVERLAP_SECONDS, PIECE_SECONDS  # noqa: E402


def _load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


torchaudio.load = _load
from audiosr import build_model  # noqa: E402


def loaded_modules():
    psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
    # 64-bit handles: declare types, or ctypes truncates them to 32-bit ints
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(wt.HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD)]
    k32.GetModuleFileNameW.argtypes = [wt.HMODULE, wt.LPWSTR, wt.DWORD]
    proc = k32.GetCurrentProcess()
    mods = (wt.HMODULE * 4096)()
    needed = wt.DWORD()
    if not psapi.EnumProcessModules(proc, mods, ctypes.sizeof(mods), ctypes.byref(needed)):
        raise OSError("EnumProcessModules failed")
    names = set()
    buf = ctypes.create_unicode_buffer(1024)
    for i in range(needed.value // ctypes.sizeof(wt.HMODULE)):
        k32.GetModuleFileNameW(mods[i], buf, 1024)
        names.add(os.path.basename(buf.value).lower())
    assert "torch_cpu.dll" in names, "module listing broken: torch_cpu.dll must be loaded"
    return names


clip = sys.argv[1]
audio, sr = engine.read_audio(clip)
audio = audio[: sr * 12]  # 2 sections, enough to exercise every code path
for checkpoint, device, half in [("basic", "cuda", speedups_half := ("sampling", "decode", "vocoder")),
                                 ("basic", "cuda", ()), ("speech", "cuda", speedups_half), ("basic", "cpu", ())]:
    model = build_model(model_name=checkpoint, device=device)
    speedups.apply(model, half if device == "cuda" else ())
    runner = AudioSRRunner(model, steps=5, guidance=2.5)
    engine.upscale(audio, sr, runner, stereo_mode="ms_full", lossy=True, chunk_seconds=PIECE_SECONDS,
                   overlap_seconds=OVERLAP_SECONDS, align=LATENT_HOP)
    del model, runner
    torch.cuda.empty_cache()
    print(f"ran {checkpoint} on {device} {'bf16' if half and device == 'cuda' else 'fp32'}", flush=True)

lib = os.path.join(os.path.dirname(torch.__file__), "lib")
used = loaded_modules()
unused = [(f, os.path.getsize(os.path.join(lib, f))) for f in os.listdir(lib)
          if f.lower().endswith(".dll") and f.lower() not in used]
print(f"\nnever loaded ({sum(s for _, s in unused) / 1024**2:.0f} MB):")
for f, s in sorted(unused, key=lambda x: -x[1]):
    print(f"  {s / 1024**2:7.1f} MB  {f}")
