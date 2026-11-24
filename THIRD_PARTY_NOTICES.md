# Third-party notices

AudioSR Upscaler is an independent project. It is not affiliated with or endorsed by the
authors of AudioSR. The Windows setup bundles the following components; each remains under
its own license.

| Component | License | Notes |
|---|---|---|
| [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) (`audiosr` 0.0.7) | MIT, Copyright (c) Haohe Liu | The AI model code. Model weights are downloaded from Hugging Face (`haoheliu/audiosr_basic`, `haoheliu/audiosr_speech`) on first use and are not redistributed here. |
| [PyTorch](https://pytorch.org), torchaudio, torchvision | BSD-3-Clause | |
| NVIDIA CUDA runtime libraries and cuDNN (inside PyTorch) | NVIDIA CUDA Toolkit EULA / cuDNN license | Redistributed as part of PyTorch's CUDA build. |
| [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause | |
| [FFmpeg](https://ffmpeg.org) (bundled by PyAV, in `av.libs`) | LGPL-3.0-or-later | Built without GPL components; dynamically linked DLLs that can be replaced. |
| [soundfile](https://github.com/bastibe/python-soundfile) / libsndfile | BSD-3-Clause / LGPL-2.1-or-later | |
| [librosa](https://librosa.org) | ISC | |
| [NumPy](https://numpy.org), [SciPy](https://scipy.org), [scikit-learn](https://scikit-learn.org) | BSD-3-Clause | |
| [Numba](https://numba.pydata.org) / llvmlite | BSD-2-Clause | |
| [Transformers](https://github.com/huggingface/transformers), [timm](https://github.com/huggingface/pytorch-image-models), huggingface_hub | Apache-2.0 | |
| [Matplotlib](https://matplotlib.org) | Matplotlib License (PSF-based) | |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | MIT | ITU-R BS.1770 loudness. |
| [Sun Valley ttk theme](https://github.com/rdbende/Sun-Valley-ttk-theme) (`sv-ttk`) | MIT | |
| Python, Tcl/Tk | PSF License, Tcl/Tk License | |
| [PyInstaller](https://pyinstaller.org) bootloader | GPL-2.0 with bootloader exception | The exception allows distributing the built app under any license. |
| [WiX Toolset](https://wixtoolset.org) 5 (installer build only) | MS-RL | Not part of the installed app apart from the setup engine. |

The hamster artwork in `assets/` was provided by the project owner for this app.

## Citation

If you use AudioSR in research, please cite the original paper:

```bibtex
@inproceedings{liu2024audiosr,
  title={{AudioSR}: Versatile audio super-resolution at scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
  pages={1076--1080},
  year={2024},
  organization={IEEE}
}
```
