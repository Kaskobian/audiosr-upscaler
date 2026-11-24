"""Runtime patches that make AudioSR faster and lighter on VRAM. Weights are untouched.

- Exact attention in query slices for the VAE (never builds the full hw x hw matrix).
- Parts that inference never touches (CLAP, EMA weights, the conditioning VAE's
  decoder + vocoder) move to system RAM: about 2.9 GB of VRAM freed.
- Optional bf16 on tensor cores per stage (sampling / VAE decode / vocoder).
"""
import functools

import torch

ATTN_SLICE_BYTES = 512 * 1024 ** 2  # score matrix memory per slice


def _sliced_vae_attention(self, x):
    h_ = self.norm(x)
    q, k, v = self.q(h_), self.k(h_), self.v(h_)
    b, c, h, w = q.shape
    n = h * w
    q = q.reshape(b, c, n).transpose(1, 2)   # b, n, c
    k = k.reshape(b, c, n)                    # b, c, n
    v = v.reshape(b, c, n).transpose(1, 2)    # b, n, c
    scale = c ** -0.5
    out = torch.empty_like(q)
    rows = max(1, ATTN_SLICE_BYTES // (b * n * q.element_size()))
    for i in range(0, n, rows):
        scores = torch.bmm(q[:, i:i + rows], k).mul_(scale).softmax(dim=-1)
        out[:, i:i + rows] = torch.bmm(scores, v)
    out = out.transpose(1, 2).reshape(b, c, h, w)
    return x + self.proj_out(out)


def patch_attention():
    from audiosr.latent_diffusion.modules.diffusionmodules import model as vae_model
    vae_model.AttnBlock.forward = _sliced_vae_attention


def slim(model):
    """Move modules inference never calls to CPU RAM. Returns GB of VRAM freed."""
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if hasattr(model, "clap"):
        model.clap.to("cpu")
    if getattr(model, "model_ema", None) is not None:
        model.model_ema.to("cpu")
    for cond in getattr(model, "cond_stage_models", []):
        vae = getattr(cond, "vae", None)
        if vae is not None:  # conditioning only calls vae.encode()
            for part in ("decoder", "vocoder", "post_quant_conv"):
                if hasattr(vae, part):
                    getattr(vae, part).to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return (before - torch.cuda.memory_allocated()) / 1024 ** 3
    return 0.0


def _autocast_float(fn, dtype):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.autocast("cuda", dtype=dtype):
            out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out.float()
        if isinstance(out, tuple):
            return tuple(o.float() if isinstance(o, torch.Tensor) and o.is_floating_point() else o for o in out)
        return out
    return wrapper


def use_half_precision(model, stages, dtype=torch.bfloat16):
    """stages: any of 'sampling', 'decode', 'vocoder'. Applies to this model instance."""
    if "sampling" in stages:
        model.sample_log = _autocast_float(model.sample_log, dtype)
    if "decode" in stages:
        model.decode_first_stage = _autocast_float(model.decode_first_stage, dtype)
    if "vocoder" in stages:
        voc = model.first_stage_model.vocoder
        voc.forward = _autocast_float(voc.forward, dtype)


def apply(model=None, half_stages=()):
    patch_attention()
    torch.backends.cudnn.benchmark = True
    freed = slim(model) if model is not None else 0.0
    if model is not None and half_stages and torch.cuda.is_available():
        use_half_precision(model, half_stages)
    return freed
