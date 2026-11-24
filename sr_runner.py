"""Runs AudioSR one piece at a time, with each piece continuing the one before it.

Plain AudioSR generates every piece from scratch, so two neighbouring pieces invent
different high frequencies in the region they share and a crossfade blends two textures.
Here the shared region of the new piece is pinned to what the previous piece generated
(diffusion inpainting through the DDIM sampler's mask/x0), so the model continues the same
material across the join.
"""
import numpy as np

LATENT_HOP = 3840          # samples at 48 kHz per latent frame (measured: 12.5 frames/s)
PIECE_SECONDS = 10.24      # 128 latent frames, AudioSR's native 2 x 5.12 s length
OVERLAP_SECONDS = 2.56     # 32 latent frames


def safe_steps(steps):
    """AudioSR's DDIM schedule indexes one past its 1000-step table when 1000 // steps divides
    999 (e.g. 3, 9, 37, 111 steps) and crashes. Nudge those to the next safe count."""
    steps = max(2, int(steps))
    while 999 % (1000 // steps) == 0:
        steps += 1
    return steps


def _dc_only(waveform):
    # AudioSR peak-normalises every file it reads, which gives each piece a different gain.
    # The engine already scales the whole track once, so only remove DC here.
    return waveform - np.mean(waveform)


class AudioSRRunner:
    def __init__(self, model, steps=50, guidance=3.5, seed=42, continuity=True, shared_gain=True,
                 edge_frames=0):
        from audiosr import pipeline, utils
        from audiosr.latent_diffusion.models.ddpm import DDPM

        self.model, self.steps, self.guidance, self.seed = model, safe_steps(steps), guidance, seed
        self.continuity = continuity
        self.edge_frames = edge_frames  # soften the pinned region's end over this many frames
        self._pipeline, self._ddpm = pipeline, DDPM
        self.prev_latent = None
        self.prev_frames = 0
        utils.normalize_wav = _dc_only if shared_gain else self._original_normalize(utils)

    @staticmethod
    def _original_normalize(utils):
        def normalize_wav(waveform):
            waveform = waveform - np.mean(waveform)
            return 0.5 * waveform / (np.max(np.abs(waveform)) + 1e-8)
        return normalize_wav

    def __call__(self, path, carry_samples=0):
        """carry_samples: how much of this piece's start overlaps the previous piece
        (0 = first piece of a track, which also resets the continuation)."""
        import torch  # imported here so the GUI can read this module's constants instantly

        m = self.model
        if carry_samples == 0:
            self.prev_latent = None
        self._pipeline.seed_everything(int(self.seed))
        batch, _ = self._pipeline.make_batch_for_super_resolution(path)
        with torch.no_grad():
            z, c = m.get_input(batch, m.first_stage_key, unconditional_prob_cfg=0.0)
            m.latent_t_size = z.size(-2)
            c = m.filter_useful_cond_dict(c)
            uc = None
            if self.guidance != 1.0:
                uc = {key: m.cond_stage_models[meta["model_idx"]].get_unconditional_condition(z.shape[0])
                      for key, meta in m.cond_stage_model_metadata.items()}

            extra = {}
            carry = carry_samples // LATENT_HOP
            if self.continuity and self.prev_latent is not None and carry > 0:
                tail = self.prev_latent[:, :, self.prev_frames - carry:self.prev_frames]
                x0 = torch.zeros_like(z)
                mask = torch.zeros_like(z)
                x0[:, :, :carry] = tail.to(z.dtype)
                mask[:, :, :carry] = 1.0
                edge = min(self.edge_frames, carry)
                if edge:
                    ramp = torch.linspace(1.0, 0.0, edge + 2, device=z.device, dtype=z.dtype)[1:-1]
                    mask[:, :, carry - edge:carry] = ramp[None, None, :, None]
                extra = dict(mask=mask, x0=x0)

            samples, _ = m.sample_log(cond=c, batch_size=z.shape[0], x_T=None, ddim=True,
                                      ddim_steps=self.steps, eta=1.0,
                                      unconditional_guidance_scale=self.guidance,
                                      unconditional_conditioning=uc, **extra)
            # Remember this piece's latent and how many frames hold real (unpadded) audio.
            self.prev_latent = samples.float()
            self.prev_frames = int(self._piece_samples(path) // LATENT_HOP)

            mel = m.decode_first_stage(samples)
            mel = m.mel_replace_ops(mel, self._ddpm.get_input(m, batch, "lowpass_mel"))
            wav = m.mel_spectrogram_to_waveform(mel, savepath="", bs=None, save=False)
            wav = m.postprocessing(wav, self._ddpm.get_input(m, batch, "waveform_lowpass"))
        peak = np.max(np.abs(wav), axis=-1, keepdims=True)
        return (0.5 * wav / np.maximum(peak, 1e-8))[0, 0]

    @staticmethod
    def _piece_samples(path):
        import soundfile as sf
        return sf.info(path).frames
