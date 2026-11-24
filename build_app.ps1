# Builds dist\AudioSR_Upscaler (the app folder the installers package).
# Usage: powershell -File build_app.ps1 [-Python path\to\venv\Scripts\python.exe]
# Then:  powershell -File installer\build.ps1 -Version 1.0.0
param([string]$Python = "python")   # the Python environment with requirements.txt installed
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = $Python
$audiosr = & $py -c "import audiosr, os; print(os.path.dirname(audiosr.__file__))"
Set-Location $root

& $py -m PyInstaller --noconfirm --onedir --windowed --name AudioSR_Upscaler `
    --icon "assets\hamster.ico" --add-data "assets;assets" `
    --collect-data audiosr --collect-submodules audiosr --add-data "$audiosr;audiosr" `
    --collect-data sv_ttk --collect-all av --collect-all librosa --collect-all soundfile `
    --collect-all pyloudnorm `
    --hidden-import engine --hidden-import speedups --hidden-import sr_runner --hidden-import remaster `
    audiosr_gui.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }

# GPU libraries that real processing never loads, measured with installer\dll_usage.py
# across GPU bf16, GPU fp32, CPU and both checkpoints. Leaving them out keeps the
# single-file setup under GitHub's 2 GiB release limit. Re-measure after a PyTorch upgrade.
# (torch loads every DLL in its folder at start-up, so "loaded" says nothing; these are the
# ones no other binary links to and that serve features this app never uses.)
$prune = @(
    "cudnn_adv64_9.dll",        # cuDNN RNN / advanced attention - model uses convs + bmm attention
    "cusolverMg64_11.dll",      # multi-GPU linear algebra
    "nvrtc64_120_0.alt.dll",    # alternate copy of the CUDA runtime compiler (primary stays)
    "nvperf_host.dll",          # GPU profiling
    "cufftw64_11.dll"           # FFTW-compatibility shim
)
$lib = Join-Path $root "dist\AudioSR_Upscaler\_internal\torch\lib"
foreach ($dll in $prune) {
    $p = Join-Path $lib $dll
    if (Test-Path $p) { Remove-Item -LiteralPath $p -Force; "pruned $dll" }
}
"app folder: {0:N2} GB" -f (((Get-ChildItem (Join-Path $root "dist\AudioSR_Upscaler") -Recurse -File |
    Measure-Object Length -Sum).Sum) / 1GB)
