"""AudioSR Upscaler: desktop GUI for AudioSR audio super-resolution."""
import os
import sys

# Windowed (no-console) builds set sys.stdout/stderr to None, and transformers
# crashes on import without them. Give them a sink before anything imports.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import ctypes
import queue
import re
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import sv_ttk

import engine
import remaster
from engine import Cancelled
from sr_runner import LATENT_HOP, OVERLAP_SECONDS, PIECE_SECONDS

APP_NAME = "AudioSR Upscaler"
# Bundled files live next to this script, or in PyInstaller's unpack folder in the .exe.
ASSETS = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "assets")
LATENT_T_PER_SECOND = 12.8
# Each model = an AudioSR checkpoint + a processing preset tuned for that material, measured
# with eval_models.py against the original AudioSR (restored-band LSD, lower = better).
# Music, 20 s stereo cut at 11 kHz: original AudioSR 12.65 dB (side 28.9 = no stereo highs);
# full-width stereo at guidance 2.5 gave 11.19 dB (side 10.1); guidance 3.5 / 5.0 were worse.
MODELS = {
    "General": dict(checkpoint="basic", stereo="ms", guidance=3.5),
    "Music": dict(checkpoint="basic", stereo="ms_full", guidance=2.5),
    # Speech, 29 s voice cut at 4 kHz: original AudioSR 9.16 dB (worse than the untouched
    # input, 7.82); speech checkpoint in this pipeline at guidance 2.5 gave 4.44 dB (3.5: 4.50,
    # 5.0: 4.66; general checkpoint 4.57). Voices sit in the centre, so only the mid is restored.
    "Speech / Voice": dict(checkpoint="speech", stereo="ms", guidance=2.5),
}
MODEL_REPOS = {"basic": "haoheliu/audiosr_basic", "speech": "haoheliu/audiosr_speech"}
DEVICE_CHOICES = {"Auto": "auto", "GPU (CUDA)": "cuda", "CPU": "cpu"}
HALF = ("sampling", "decode", "vocoder")
# (bf16 stages, piece length in seconds; None = as long as VRAM allows).
# Measured on a 30 s music test: 10.24 s pieces restored the highs best (AudioSR was trained
# on short clips); long pieces were ~1.7x faster. bf16 drift was ~7x smaller than a seed change.
# The whole song is always processed; these only set how big a section the AI handles at once.
MODES = {"Best quality": (HALF, PIECE_SECONDS),
         "Faster (bigger sections)": (HALF, None),
         "Exact fp32 (slowest)": ((), PIECE_SECONDS)}
UNIT_SECONDS = 5.12  # AudioSR pads every piece up to a multiple of this
# Share of one model run spent in prep and sampling (rest is decoding), measured on an
# RTX 5070 Ti. Only used to place the progress bar between step updates.
PREP_SHARE, SAMPLING_SHARE = 0.14, 0.35
# Longest clip sent to the GPU in one piece, per GB of VRAM left after the model loads.
# Measured on 12 GB: bf16 used ~6.6 GB for 53 s (~8 s/GB); fp32 used ~7.8 GB for 53 s
# with a much larger sampling peak. Values include a safety margin.
SECONDS_PER_FREE_GB = {"half": 7.0, "full": 4.0}
# Seam test (30 s music cut at 11 kHz, 2 seeds, full-length last piece): continuation cut the
# high-band disagreement between neighbouring pieces from ~7.5 dB to ~3.1 dB; fidelity to the
# true original was within noise (13.8 vs 13.6 dB LSD; seed spread ~2 dB). Softening the
# pinned region's edge made no difference.
CONTINUATION_EDGE_FRAMES = 0
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# --- Hardware info that needs no torch, so it shows instantly ------------------
def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        import platform
        return platform.processor() or "Unknown CPU"


def total_ram_gb():
    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemStatus()
    status.dwLength = ctypes.sizeof(status)
    try:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1024 ** 3
    except (AttributeError, OSError):
        return None


def windows_uses_dark_mode():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def set_dark_title_bar(root, dark):
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
    except (AttributeError, OSError):
        pass


def fmt_time(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class QueueWriter:
    """Sends print() output from audiosr to the details log."""

    def __init__(self, app):
        self.app = app

    def write(self, text):
        text = ANSI.sub("", text or "").strip()
        if text:
            self.app.post("log", text)

    def flush(self):
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.engine = None          # torch/audiosr handles, filled by the loader thread
        self.model = None
        self.model_key = None
        self.free_vram_gb = None
        self.busy = False
        self.cancel_event = threading.Event()
        self.started_at = None
        self.runs_started_at = None
        self.run_started_at = None
        self.last_run_seconds = None
        self.run = (0, 1)
        self.fraction = 0.0
        self.last_output = None
        self.input_info = None      # (duration, sample_rate, channels)
        self.dark = windows_uses_dark_mode()

        root.title(APP_NAME)
        root.minsize(760, 700)
        sv_ttk.set_theme("dark" if self.dark else "light")
        set_dark_title_bar(root, self.dark)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20))
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#8a8a8a")
        style.configure("Key.TLabel", font=("Segoe UI", 10), foreground="#8a8a8a")
        style.configure("Val.TLabel", font=("Segoe UI Semibold", 10))
        style.configure("Stage.TLabel", font=("Segoe UI Semibold", 11))
        style.configure("Warn.TLabel", foreground="#e8a33d")
        style.configure("Good.TLabel", foreground="#4cc26b")

        self.build_ui()
        self.root.after(100, self.poll)
        threading.Thread(target=self.load_engine, daemon=True).start()

    # --- layout ----------------------------------------------------------------
    def build_ui(self):
        outer = ttk.Frame(self.root, padding=(20, 16, 20, 16))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        self.outer = outer

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, rowspan=2, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        self.mascot_frames = []
        for i in range(3):
            try:
                self.mascot_frames.append(tk.PhotoImage(file=os.path.join(ASSETS, f"hamster_{i}.png")))
            except tk.TclError:
                break
        self.mascot_index = 0
        self.mascot_tick = 0
        self.mascot = ttk.Label(header, image=self.mascot_frames[0] if self.mascot_frames else "")
        self.mascot.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ttk.Label(header, text=APP_NAME, style="Title.TLabel").grid(row=0, column=1, sticky="sw")
        ttk.Label(header, text="AI super-resolution and reference remastering for music and speech",
                  style="Sub.TLabel").grid(row=1, column=1, sticky="nw")

        # Hardware card
        hw = ttk.LabelFrame(outer, text="Hardware", padding=(14, 8))
        hw.grid(row=2, column=0, sticky="ew")
        hw.columnconfigure(1, weight=1)
        ram = total_ram_gb()
        self.hw_vars = {}
        rows = [("Runs on", "Starting engine..."), ("GPU", "Detecting..."), ("VRAM", "-"),
                ("CPU", f"{cpu_name()} ({os.cpu_count()} threads)"),
                ("RAM", f"{ram:.0f} GB" if ram else "Unknown"), ("Engine", "Loading PyTorch...")]
        for r, (key, value) in enumerate(rows):
            ttk.Label(hw, text=key, style="Key.TLabel", width=9).grid(row=r, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=value)
            label = ttk.Label(hw, textvariable=var, style="Val.TLabel")
            label.grid(row=r, column=1, sticky="w", pady=1)
            self.hw_vars[key] = (var, label)

        # Files card
        files = ttk.LabelFrame(outer, text="Files", padding=(14, 8))
        files.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        files.columnconfigure(1, weight=1)
        self.input_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads"))
        self.name_var = tk.StringVar()
        self.info_var = tk.StringVar(value="WAV, FLAC, MP3 or M4A")
        self.format_var = tk.StringVar(value=next(iter(engine.OUTPUT_FORMATS)))
        self.stereo_var = tk.StringVar(value=next(iter(engine.STEREO_MODES)))
        self.ext_var = tk.StringVar(value=".wav")

        ttk.Label(files, text="Input", width=9).grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.input_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=8, pady=3)
        ttk.Button(files, text="Browse...", command=self.pick_input).grid(row=0, column=3)
        self.info_label = ttk.Label(files, textvariable=self.info_var, style="Key.TLabel")
        self.info_label.grid(row=1, column=1, columnspan=3, sticky="w", padx=8, pady=(0, 6))

        ttk.Label(files, text="Save to", width=9).grid(row=2, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.outdir_var).grid(row=2, column=1, columnspan=2, sticky="ew", padx=8, pady=3)
        ttk.Button(files, text="Browse...", command=self.pick_outdir).grid(row=2, column=3)

        ttk.Label(files, text="File name", width=9).grid(row=3, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.name_var).grid(row=3, column=1, sticky="ew", padx=(8, 4), pady=3)
        ttk.Label(files, textvariable=self.ext_var, style="Key.TLabel", width=5).grid(row=3, column=2, sticky="w")

        ttk.Label(files, text="Format", width=9).grid(row=4, column=0, sticky="w")
        out_row = ttk.Frame(files)
        out_row.grid(row=4, column=1, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Combobox(out_row, textvariable=self.format_var, values=list(engine.OUTPUT_FORMATS),
                     state="readonly", width=20).pack(side="left")
        ttk.Label(out_row, text="Channels", style="Key.TLabel").pack(side="left", padx=(16, 6))
        self.stereo_box = ttk.Combobox(out_row, textvariable=self.stereo_var, values=list(engine.STEREO_MODES),
                                       state="readonly", width=28)
        self.stereo_box.pack(side="left")
        self.format_var.trace_add("write", lambda *_: self.ext_var.set(engine.OUTPUT_FORMATS[self.format_var.get()]))

        # Reference remaster (optional)
        ttk.Separator(files).grid(row=5, column=0, columnspan=4, sticky="ew", pady=(10, 8))
        self.ref_var = tk.StringVar()
        self.ref_info_var = tk.StringVar(value="Optional: a track whose sound you want. Leave empty to only upscale.")
        self.strength_var = tk.IntVar(value=80)
        self.loudness_var = tk.BooleanVar(value=True)
        ttk.Label(files, text="Reference", width=9).grid(row=6, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.ref_var).grid(row=6, column=1, columnspan=2, sticky="ew", padx=8, pady=3)
        ref_buttons = ttk.Frame(files)
        ref_buttons.grid(row=6, column=3)
        ttk.Button(ref_buttons, text="Browse...", command=self.pick_reference).pack(side="left")
        ttk.Button(ref_buttons, text="Clear", command=lambda: (self.ref_var.set(""), self.refresh_reference())
                   ).pack(side="left", padx=(4, 0))
        ttk.Label(files, textvariable=self.ref_info_var, style="Key.TLabel").grid(
            row=7, column=1, columnspan=3, sticky="w", padx=8, pady=(0, 4))
        match_row = ttk.Frame(files)
        match_row.grid(row=8, column=1, columnspan=3, sticky="w", padx=8, pady=3)
        ttk.Label(match_row, text="Match strength", style="Key.TLabel").pack(side="left")
        self.strength_scale = ttk.Scale(match_row, from_=0, to=100, length=180,
                                        command=lambda v: self.strength_var.set(int(float(v))))
        self.strength_scale.set(self.strength_var.get())
        self.strength_scale.pack(side="left", padx=8)
        ttk.Label(match_row, textvariable=self.strength_var, width=3).pack(side="left")
        ttk.Label(match_row, text="%", style="Key.TLabel").pack(side="left", padx=(0, 16))
        self.loudness_check = ttk.Checkbutton(match_row, text="Match loudness", variable=self.loudness_var)
        self.loudness_check.pack(side="left")
        self.match_widgets = [self.strength_scale, self.loudness_check]
        self.ref_var.trace_add("write", lambda *_: self.refresh_reference())
        self.refresh_reference()

        # Settings card
        settings = ttk.LabelFrame(outer, text="Settings", padding=(14, 8))
        settings.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.model_var = tk.StringVar(value="Music")
        self.device_var = tk.StringVar(value="Auto")
        self.speed_var = tk.StringVar(value=next(iter(MODES)))
        self.steps_var = tk.IntVar(value=50)
        self.guidance_var = tk.DoubleVar(value=3.5)
        self.seed_var = tk.IntVar(value=42)

        def field(row, col, text, widget):
            ttk.Label(settings, text=text, style="Key.TLabel").grid(row=row * 2, column=col, sticky="w", padx=(0, 4))
            widget.grid(row=row * 2 + 1, column=col, sticky="w", padx=(0, 16), pady=(0, 6))

        field(0, 0, "Model", ttk.Combobox(settings, textvariable=self.model_var, values=list(MODELS),
                                          state="readonly", width=15))
        self.device_box = ttk.Combobox(settings, textvariable=self.device_var, values=list(DEVICE_CHOICES),
                                       state="readonly", width=11)
        field(0, 1, "Device", self.device_box)
        self.speed_box = ttk.Combobox(settings, textvariable=self.speed_var, values=list(MODES),
                                      state="readonly", width=26)
        field(0, 2, "Mode", self.speed_box)
        field(1, 0, "Steps", ttk.Spinbox(settings, from_=10, to=200, textvariable=self.steps_var, width=6))
        field(1, 1, "Guidance", ttk.Spinbox(settings, from_=1.0, to=10.0, increment=0.5,
                                            textvariable=self.guidance_var, width=6))
        field(1, 2, "Seed", ttk.Spinbox(settings, from_=0, to=999999, textvariable=self.seed_var, width=8))
        self.seamless_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="Seamless joins: each piece continues the previous one's generated "
                                       "highs (smoother, slightly less literal)",
                        variable=self.seamless_var).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        for var in (self.device_var, self.speed_var):
            var.trace_add("write", lambda *_: self.refresh_device())
        self.model_var.trace_add("write", lambda *_: self.apply_model_preset())
        self.apply_model_preset()

        # Progress card
        prog = ttk.LabelFrame(outer, text="Progress", padding=(14, 8))
        prog.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        prog.columnconfigure(0, weight=1)
        self.stage_var = tk.StringVar(value="Waiting for engine...")
        self.detail_var = tk.StringVar(value="")
        self.time_var = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self.stage_var, style="Stage.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(prog, textvariable=self.time_var, style="Key.TLabel").grid(row=0, column=1, sticky="e")
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(prog, textvariable=self.detail_var, style="Key.TLabel").grid(row=2, column=0, sticky="w")
        self.percent_var = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self.percent_var, style="Val.TLabel").grid(row=2, column=1, sticky="e")

        # Buttons
        buttons = ttk.Frame(outer)
        buttons.grid(row=6, column=0, sticky="ew", pady=(14, 0))
        self.start_btn = ttk.Button(buttons, text="Upscale", style="Accent.TButton", width=14,
                                    command=self.start)
        self.start_btn.pack(side="left")
        self.start_btn.state(["disabled"])
        self.cancel_btn = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.cancel_btn.pack(side="left", padx=8)
        self.cancel_btn.state(["disabled"])
        self.open_btn = ttk.Button(buttons, text="Open output folder", command=self.open_output)
        self.open_btn.pack(side="left")
        self.open_btn.state(["disabled"])
        self.log_btn = ttk.Button(buttons, text="Show details", command=self.toggle_log)
        self.log_btn.pack(side="right")

        # Details log (hidden until asked for)
        self.log_frame = ttk.Frame(outer)
        self.log_frame.columnconfigure(0, weight=1)
        self.log_frame.rowconfigure(0, weight=1)
        bg, fg = ("#1c1c1c", "#d6d6d6") if self.dark else ("#fafafa", "#202020")
        self.log = tk.Text(self.log_frame, height=10, wrap="word", state="disabled", relief="flat",
                           bg=bg, fg=fg, insertbackground=fg, font=("Consolas", 9), padx=8, pady=6)
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.log_frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    def toggle_log(self):
        if self.log_frame.winfo_ismapped():
            self.log_frame.grid_forget()
            self.outer.rowconfigure(7, weight=0)
            self.log_btn.configure(text="Show details")
        else:
            self.log_frame.grid(row=7, column=0, sticky="nsew", pady=(12, 0))
            self.outer.rowconfigure(7, weight=1)
            self.log_btn.configure(text="Hide details")

    # --- engine (heavy imports off the UI thread) -------------------------------
    def load_engine(self):
        try:
            import torch
            import torchaudio

            # torchaudio's loader needs torchcodec, which fails to load on some machines.
            # AudioSR only reads the temporary WAV chunks the engine writes, so soundfile is enough.
            def load_with_soundfile(path, *args, **kwargs):
                data, sr = engine.sf.read(str(path), dtype="float32", always_2d=True)
                return torch.from_numpy(data.T.copy()), sr

            torchaudio.load = load_with_soundfile

            from audiosr import build_model, super_resolution
            import audiosr.latent_diffusion.models.ddim as ddim
            from huggingface_hub import try_to_load_from_cache
            import speedups
            from sr_runner import AudioSRRunner

            ddim.tqdm = self.make_step_reporter()
            speedups.patch_attention()
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

            info = {"torch_version": torch.__version__, "cuda": torch.cuda.is_available()}
            if info["cuda"]:
                props = torch.cuda.get_device_properties(0)
                info.update(gpu=props.name, vram=props.total_memory / 1024 ** 3,
                            cuda_version=torch.version.cuda, bf16=torch.cuda.is_bf16_supported())
            self.engine = dict(torch=torch, build_model=build_model, super_resolution=super_resolution,
                               cached=try_to_load_from_cache, speedups=speedups, runner=AudioSRRunner, **info)
            self.post("engine_ready")
        except Exception:
            self.post("engine_failed", traceback.format_exc())

    def make_step_reporter(self):
        app = self

        class StepReporter:
            """Stands in for tqdm in AudioSR's sampler to report each step and allow cancelling."""

            def __init__(self, iterable, total=None, **_):
                self.iterable = iterable
                self.total = total if total is not None else len(iterable)

            def __len__(self):
                return self.total

            def __iter__(self):
                app.post("sampling", 0, self.total)
                for i, item in enumerate(self.iterable):
                    if app.cancel_event.is_set():
                        raise Cancelled()
                    yield item
                    app.post("sampling", i + 1, self.total)
                app.post("decoding")

        return StepReporter

    # --- helpers ------------------------------------------------------------------
    def post(self, kind, *args):
        self.events.put((kind, *args))

    def resolved_device(self):
        choice = DEVICE_CHOICES[self.device_var.get()]
        if not self.engine:
            return choice
        if choice == "auto":
            return "cuda" if self.engine["cuda"] else "cpu"
        return choice

    def half_stages(self):
        if self.resolved_device() != "cuda" or not self.engine.get("bf16"):
            return ()
        return MODES[self.speed_var.get()][0]

    def chunk_seconds(self):
        """Piece length sent to the model. Longer files stream through in pieces while the
        full track stays in system RAM, so file length is never limited by VRAM."""
        preset = MODES[self.speed_var.get()][1]
        if preset:
            return preset
        if self.resolved_device() != "cuda":
            return 30.0  # keeps CPU RAM use and progress granularity reasonable
        # Before the model loads, estimate what will be free (measured: ~4.8 GB in use after loading).
        free = self.free_vram_gb if self.free_vram_gb is not None else self.engine["vram"] - 4.8
        per_gb = SECONDS_PER_FREE_GB["half" if self.half_stages() else "full"]
        # whole 5.12 s units: AudioSR pads to them anyway, and it keeps pieces on the latent grid
        return max(PIECE_SECONDS, (free * per_gb) // UNIT_SECONDS * UNIT_SECONDS)

    def refresh_device(self):
        if not self.engine:
            return
        var, label = self.hw_vars["Runs on"]
        if self.resolved_device() == "cuda":
            mode = "bf16 tensor cores" if self.half_stages() else "fp32"
            var.set(f"GPU  ·  {self.engine['gpu']}  ·  {mode}")
            label.configure(style="Good.TLabel")
        else:
            var.set(f"CPU  ·  {os.cpu_count()} threads  ·  much slower than a GPU")
            label.configure(style="Warn.TLabel" if self.engine["cuda"] else "Val.TLabel")
        self.refresh_input_info()

    def refresh_input_info(self):
        path = self.input_var.get().strip()
        self.input_info = None
        if not path:
            self.info_var.set("WAV, FLAC, MP3 or M4A")
            self.info_label.configure(style="Key.TLabel")
            return
        try:
            duration, sr, channels = engine.audio_info(path)
        except Exception:
            self.info_var.set("Can't read this file as audio")
            self.info_label.configure(style="Warn.TLabel")
            return
        self.input_info = (duration, sr, channels)
        ext = os.path.splitext(path)[1].lower()
        kind = ext.lstrip(".").upper() + (" (lossy)" if ext in engine.LOSSY_EXTS else "")
        ch = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels} ch"
        text = f"{fmt_time(duration)}  ·  {sr / 1000:g} kHz  ·  {ch}  ·  {kind}"
        self.stereo_box.state(["disabled"] if channels == 1 else ["!disabled"])
        if self.engine:
            chunk = self.chunk_seconds()
            sections = len(engine.plan_chunks(int(duration * engine.OUT_SR), int(chunk * engine.OUT_SR),
                                              int(OVERLAP_SECONDS * engine.OUT_SR), LATENT_HOP))
            text += ("  ·  whole track in one pass" if sections == 1 else
                     f"  ·  whole track, AI works through it in {sections} overlapping sections")
        self.info_var.set(text)
        self.info_label.configure(style="Key.TLabel")

    def pick_input(self):
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a"), ("All files", "*.*")])
        if path:
            self.input_var.set(path)
            self.name_var.set(os.path.splitext(os.path.basename(path))[0] + "_AudioSR")
            self.refresh_input_info()

    def apply_model_preset(self):
        """Picking a model loads its tuned defaults; the controls stay editable."""
        preset = MODELS[self.model_var.get()]
        label = next(k for k, v in engine.STEREO_MODES.items() if v == preset["stereo"])
        self.stereo_var.set(label)
        self.guidance_var.set(preset["guidance"])
        self.refresh_input_info()

    def pick_reference(self):
        path = filedialog.askopenfilename(
            title="Choose a reference track (the sound you want)",
            filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a"), ("All files", "*.*")])
        if path:
            self.ref_var.set(path)

    def refresh_reference(self):
        path = self.ref_var.get().strip()
        enabled = False
        if not path:
            self.ref_info_var.set("Optional: a track whose sound you want. Leave empty to only upscale.")
        else:
            try:
                duration, sr, channels = engine.audio_info(path)
                ch = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels} ch"
                self.ref_info_var.set(f"{fmt_time(duration)}  ·  {sr / 1000:g} kHz  ·  {ch}  ·  will match its "
                                      f"tonal balance, stereo width and loudness")
                enabled = True
            except Exception:
                self.ref_info_var.set("Can't read this reference file as audio")
        for widget in self.match_widgets:
            widget.state(["!disabled"] if enabled else ["disabled"])

    def pick_outdir(self):
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.outdir_var.set(path)

    def open_output(self):
        if self.last_output and os.path.exists(self.last_output):
            os.startfile(os.path.dirname(self.last_output))

    def write_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # --- run -----------------------------------------------------------------------
    def start(self):
        if self.busy or not self.engine:
            return
        self.refresh_input_info()
        in_path = self.input_var.get().strip()
        out_dir = self.outdir_var.get().strip()
        if not self.input_info:
            messagebox.showerror(APP_NAME, "Choose a WAV, FLAC, MP3 or M4A file first.")
            return
        if not os.path.isdir(out_dir):
            messagebox.showerror(APP_NAME, "The output folder does not exist.")
            return
        ext = engine.OUTPUT_FORMATS[self.format_var.get()]
        name = self.name_var.get().strip()
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
        if not name:
            name = os.path.splitext(os.path.basename(in_path))[0] + "_AudioSR"
        out_path = os.path.join(out_dir, name + ext)
        if os.path.abspath(out_path) == os.path.abspath(in_path):
            messagebox.showerror(APP_NAME, "The output would overwrite the input. Pick another name.")
            return
        reference = self.ref_var.get().strip()
        if reference:
            try:
                engine.audio_info(reference)
            except Exception:
                messagebox.showerror(APP_NAME, "The reference file can't be read. Pick another or clear it.")
                return
        if os.path.exists(out_path) and not messagebox.askyesno(
                APP_NAME, f"{os.path.basename(out_path)} already exists. Replace it?"):
            return
        try:
            job = dict(input=in_path, output=out_path, duration=self.input_info[0],
                       reference=reference, strength=self.strength_var.get() / 100,
                       match_loudness=bool(self.loudness_var.get()), seamless=bool(self.seamless_var.get()),
                       model=MODELS[self.model_var.get()]["checkpoint"], device=self.resolved_device(),
                       half=self.half_stages(), stereo=engine.STEREO_MODES[self.stereo_var.get()],
                       lossy=os.path.splitext(in_path)[1].lower() in engine.LOSSY_EXTS,
                       steps=int(self.steps_var.get()), guidance=float(self.guidance_var.get()),
                       seed=int(self.seed_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror(APP_NAME, "Steps, guidance and seed must be numbers.")
            return
        if job["steps"] < 1:
            messagebox.showerror(APP_NAME, "Steps must be at least 1.")
            return

        self.busy = True
        self.cancel_event.clear()
        self.started_at = time.monotonic()
        self.runs_started_at = None
        self.run_started_at = None
        self.last_run_seconds = None
        self.run = (0, 1)
        self.fraction = 0.0
        self.progress["value"] = 0
        self.percent_var.set("")
        self.start_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.open_btn.state(["disabled"])
        self.detail_var.set("")
        self.write_log(f"--- {os.path.basename(in_path)} -> {out_path}")
        threading.Thread(target=self.worker, args=(job,), daemon=True).start()

    def cancel(self):
        if self.busy:
            self.cancel_event.set()
            self.cancel_btn.state(["disabled"])
            self.stage_var.set("Cancelling...")
            self.detail_var.set("Stops at the next step")

    def ensure_model(self, job):
        torch, speedups = self.engine["torch"], self.engine["speedups"]
        key = (job["model"], job["device"], job["half"])
        if self.model_key == key:
            return
        self.model = self.model_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cached = isinstance(self.engine["cached"](MODEL_REPOS[job["model"]], "pytorch_model.bin"), str)
        self.post("stage", "Loading model..." if cached else "Downloading model (first run only)...",
                  f"{job['model']} model on {job['device'].upper()}")
        if job["device"] == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        model = self.engine["build_model"](model_name=job["model"], device=job["device"])
        freed = speedups.slim(model)
        if job["half"]:
            speedups.use_half_precision(model, job["half"])
        if job["device"] == "cuda":
            free, _ = torch.cuda.mem_get_info()
            self.free_vram_gb = free / 1024 ** 3
            self.post("log", f"Moved {freed:.1f} GB of unused model parts to system RAM; "
                             f"{self.free_vram_gb:.1f} GB VRAM free for audio")
        self.model, self.model_key = model, key

    def worker(self, job):
        torch = self.engine["torch"]
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = QueueWriter(self)
        try:
            self.ensure_model(job)
            if self.cancel_event.is_set():
                raise Cancelled()
            self.post("stage", "Reading audio...", os.path.basename(job["input"]))
            audio, sr = engine.read_audio(job["input"])
            bandwidth = engine.detect_bandwidth(audio.mean(axis=1), sr)
            if bandwidth >= engine.OUT_SR / 2 - 200:
                self.post("log", "Source already has content up to 24 kHz; skipping AudioSR.")
            else:
                self.post("log", f"Source has real content up to {bandwidth / 1000:.1f} kHz; "
                                 f"AudioSR fills {bandwidth / 1000:.1f}-24 kHz")

            # Each piece continues the previous one's generated highs, so joins stay seamless.
            run_model = self.engine["runner"](self.model, steps=job["steps"], guidance=job["guidance"],
                                              seed=job["seed"], continuity=job["seamless"],
                                              edge_frames=CONTINUATION_EDGE_FRAMES)
            chunk = self.chunk_seconds()
            while True:
                try:
                    result, info = engine.upscale(
                        audio, sr, run_model, stereo_mode=job["stereo"], lossy=job["lossy"],
                        chunk_seconds=chunk, overlap_seconds=OVERLAP_SECONDS, bandwidth=bandwidth,
                        align=LATENT_HOP,
                        on_progress=lambda i, n, label: self.post("run", i, n, label),
                        should_cancel=self.cancel_event.is_set)
                    break
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    smaller = (min(chunk, job["duration"]) / 2) // UNIT_SECONDS * UNIT_SECONDS
                    if smaller < UNIT_SECONDS:
                        raise
                    self.post("log", f"GPU ran out of memory; retrying with {smaller:.2f} s sections")
                    chunk = smaller

            if job["reference"]:
                if self.cancel_event.is_set():
                    raise Cancelled()
                self.post("stage", "Remastering to the reference...",
                          "Matching tonal balance, stereo width and loudness")
                ref, ref_sr = engine.read_audio(job["reference"])
                ref_bw = engine.detect_bandwidth(ref.mean(axis=1), ref_sr)
                result, info["remaster"] = remaster.match(
                    result, engine.to_48k(ref, ref_sr), strength=job["strength"],
                    match_loudness=job["match_loudness"], limit_hz=min(ref_bw, engine.OUT_SR / 2))
            self.post("stage", "Saving...", job["output"])
            info["gain_db"] = engine.write_audio(job["output"], result)
            self.post("done", job["output"], info)
        except Cancelled:
            self.post("cancelled")
        except torch.OutOfMemoryError:
            self.model = self.model_key = None
            torch.cuda.empty_cache()
            self.post("oom")
        except Exception:
            self.post("error", traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    # --- UI-thread event pump ------------------------------------------------------
    def poll(self):
        try:
            while True:
                kind, *args = self.events.get_nowait()
                getattr(self, "on_" + kind)(*args)
        except queue.Empty:
            pass
        if self.busy and self.started_at:
            now = time.monotonic()
            text = f"Elapsed {fmt_time(now - self.started_at)}"
            index, count = self.run
            if self.last_run_seconds:
                # The first run includes GPU warm-up, so later runs are better predictors.
                within = self.fraction * count - index
                left = self.last_run_seconds * (count - index - within)
                text += f"  ·  about {fmt_time(left)} left"
            elif self.runs_started_at and self.fraction > 0.03:
                spent = now - self.runs_started_at
                text += f"  ·  about {fmt_time(spent * (1 - self.fraction) / self.fraction)} left"
            self.time_var.set(text)
        # the hamster always munches; faster while a song is being processed
        self.mascot_tick += 1
        if self.mascot_frames and self.mascot_tick % (1 if self.busy else 2) == 0:
            self.mascot_index = (self.mascot_index + 1) % len(self.mascot_frames)
            self.mascot.configure(image=self.mascot_frames[self.mascot_index])
        self.root.after(100, self.poll)

    def set_indeterminate(self, on):
        if on and str(self.progress["mode"]) != "indeterminate":
            self.progress.configure(mode="indeterminate")
            self.progress.start(15)
        elif not on and str(self.progress["mode"]) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")

    def set_fraction(self, within_run):
        index, count = self.run
        self.fraction = min(1.0, (index + within_run) / count)
        self.set_indeterminate(False)
        self.progress["value"] = 100 * self.fraction
        self.percent_var.set(f"{100 * self.fraction:.0f}%")

    def on_log(self, text):
        self.write_log(text)

    def on_engine_ready(self):
        e = self.engine
        self.hw_vars["Engine"][0].set(f"PyTorch {e['torch_version']}"
                                      + (f"  ·  CUDA {e['cuda_version']}" if e["cuda"] else ""))
        if e["cuda"]:
            self.hw_vars["GPU"][0].set(e["gpu"])
            self.hw_vars["VRAM"][0].set(f"{e['vram']:.1f} GB  ·  unused model parts are kept in system RAM")
        else:
            self.hw_vars["GPU"][0].set("No CUDA GPU found")
            self.hw_vars["VRAM"][0].set("-")
            self.device_box.configure(values=["Auto", "CPU"])
        self.refresh_device()
        self.stage_var.set("Ready")
        self.start_btn.state(["!disabled"])

    def on_engine_failed(self, tb):
        self.hw_vars["Engine"][0].set("Failed to load. See details.")
        self.stage_var.set("Engine failed to load")
        self.write_log(tb)
        if not self.log_frame.winfo_ismapped():
            self.toggle_log()

    def on_stage(self, title, detail=""):
        self.stage_var.set(title)
        self.detail_var.set(detail)
        if not self.runs_started_at:
            self.set_indeterminate(True)

    def on_run(self, index, count, label):
        now = time.monotonic()
        if self.runs_started_at is None:
            self.runs_started_at = now
        if index > 0 and self.run_started_at:
            self.last_run_seconds = now - self.run_started_at
        self.run_started_at = now
        self.run = (index, count)
        self.run_label = label
        self.stage_var.set("Upscaling" + (f"  ·  {label}" if label else ""))
        self.detail_var.set("Preparing")
        self.set_fraction(0)

    def on_sampling(self, done, total):
        self.detail_var.set(f"Generating high frequencies  ·  step {done} of {total}")
        self.set_fraction(PREP_SHARE + SAMPLING_SHARE * done / total)

    def on_decoding(self):
        self.detail_var.set("Decoding to audio (vocoder)")
        self.set_fraction(PREP_SHARE + SAMPLING_SHARE)

    def finish(self, stage, detail="", value=0):
        self.busy = False
        self.set_indeterminate(False)
        self.progress["value"] = value
        self.percent_var.set(f"{value:.0f}%" if value else "")
        self.stage_var.set(stage)
        self.detail_var.set(detail)
        self.start_btn.state(["!disabled"])
        self.cancel_btn.state(["disabled"])

    def on_done(self, path, info):
        took = fmt_time(time.monotonic() - self.started_at)
        self.last_output = path
        self.finish("Done", f"Saved {os.path.basename(path)} in {took}", 100)
        self.open_btn.state(["!disabled"])
        self.write_log(f"Saved: {path}  ({info['runs']} model runs, {info['chunks']} section(s), "
                       f"{info['stereo_mode']})")
        if info["runs"] == 0:
            self.detail_var.set(f"Source already has content up to 24 kHz, so AI upscaling was skipped "
                                f"(nothing missing to restore)  ·  saved in {took}")
        rep = info.get("remaster")
        if rep:
            eq = ", ".join(f"{name} {gain:+.1f} dB" for name, gain in rep["eq"].items())
            width = f"  ·  width x{rep['width']:.2f}" if "width" in rep else ""
            self.detail_var.set(f"Remastered: {rep['lufs_after']:.1f} LUFS (reference {rep['lufs_reference']:.1f})"
                                f"{width}  ·  saved in {took}")
            self.write_log(f"Remaster: EQ {eq}{width}; loudness {rep['lufs_before']:.1f} -> "
                           f"{rep['lufs_after']:.1f} LUFS (reference {rep['lufs_reference']:.1f}); "
                           f"limiter caught up to {rep['limiter_db']:.1f} dB of peaks")
        if info["gain_db"] < -0.05:
            self.write_log(f"Lowered the level by {-info['gain_db']:.1f} dB so the restored highs don't clip.")

    def on_cancelled(self):
        self.finish("Cancelled", "Nothing was saved")
        self.write_log("Cancelled.")

    def on_oom(self):
        self.finish("Out of GPU memory", "Even small sections didn't fit. Set Device to CPU.")
        self.write_log("CUDA out of memory.")
        messagebox.showwarning(APP_NAME, "The GPU ran out of memory even with small sections.\n\n"
                                         "Close other GPU-heavy apps, or set Device to CPU (much slower).")

    def on_error(self, tb):
        self.finish("Failed", "See details for the error")
        self.write_log("ERROR:\n" + tb)
        if not self.log_frame.winfo_ismapped():
            self.toggle_log()
        messagebox.showerror(APP_NAME, "Processing failed. The details panel shows the error.")

    def on_close(self):
        if self.busy and not messagebox.askyesno(APP_NAME, "Processing is still running. Quit anyway?"):
            return
        self.root.destroy()


def run_batch(argv):
    """AudioSR_Upscaler.exe --batch IN OUT [--model Music] [--steps 50] [--log FILE]

    Processes one file through exactly the same code as the window, without showing it.
    Used to verify release builds; the exit code is 0 only if the file was written."""
    import argparse
    p = argparse.ArgumentParser(prog="AudioSR_Upscaler --batch")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", default="Music", choices=list(MODELS))
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--mode", default=next(iter(MODES)), choices=list(MODES))
    p.add_argument("--device", default="Auto", choices=list(DEVICE_CHOICES))
    p.add_argument("--log", default=None)
    a = p.parse_args(argv)
    ext = os.path.splitext(a.output)[1].lower()
    fmt = next((k for k, v in engine.OUTPUT_FORMATS.items() if v == ext), None)
    if fmt is None:
        raise SystemExit(f"Unsupported output extension: {ext}")

    root = tk.Tk()
    root.withdraw()
    messagebox.askyesno = lambda *x, **k: True          # replace an existing output
    messagebox.showerror = messagebox.showwarning = lambda *x, **k: None
    app = App(root)
    state = {"started": False, "ok": False}

    def tick():
        if not state["started"] and app.engine:
            app.input_var.set(os.path.abspath(a.input))
            app.outdir_var.set(os.path.dirname(os.path.abspath(a.output)))
            app.name_var.set(os.path.splitext(os.path.basename(a.output))[0])
            app.format_var.set(fmt)
            app.model_var.set(a.model)
            app.speed_var.set(a.mode)
            app.device_var.set(a.device)
            app.steps_var.set(a.steps)
            app.start()
            state["started"] = True
        elif state["started"] and not app.busy:
            state["ok"] = app.stage_var.get() == "Done" and os.path.exists(a.output)
            if a.log:
                with open(a.log, "w", encoding="utf-8") as fh:
                    fh.write(f"{app.stage_var.get()} | {app.detail_var.get()}\n")
                    fh.write(app.log.get("1.0", "end"))
            root.destroy()
            return
        elif not app.engine and "failed" in app.stage_var.get().lower():
            if a.log:
                with open(a.log, "w", encoding="utf-8") as fh:
                    fh.write(app.log.get("1.0", "end"))
            root.destroy()
            return
        root.after(250, tick)

    root.after(250, tick)
    root.mainloop()
    sys.exit(0 if state["ok"] else 1)


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on high-DPI screens
    except (AttributeError, OSError):
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch(sys.argv[2:])
        return
    try:  # own taskbar entry and icon instead of Python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AudioSR.Upscaler")
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    try:
        root.iconbitmap(default=os.path.join(ASSETS, "hamster.ico"))
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
