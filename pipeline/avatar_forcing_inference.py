from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from utils.inject import slice_conditional_dict
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass(frozen=True)
class _ProfileHandles:
    init_start: torch.cuda.Event
    init_end: torch.cuda.Event
    diffusion_start: torch.cuda.Event
    diffusion_end: torch.cuda.Event
    vae_start: torch.cuda.Event
    vae_end: torch.cuda.Event
    block_start: torch.cuda.Event
    block_end: torch.cuda.Event
    block_times: List[float]


def _make_profile(profile: bool) -> Optional[_ProfileHandles]:
    if not profile:
        return None
    evt = lambda: torch.cuda.Event(enable_timing=True)
    return _ProfileHandles(
        init_start=evt(),
        init_end=evt(),
        diffusion_start=evt(),
        diffusion_end=evt(),
        vae_start=evt(),
        vae_end=evt(),
        block_start=evt(),
        block_end=evt(),
        block_times=[],
    )


class AvatarForcingInferencePipeline(torch.nn.Module):
    """
    AvatarForcing inference pipeline.

    Notes:
    - For backward compatibility, `CausalInferencePipeline` is kept as an alias.
    """

    def __init__(
        self,
        args,
        device,
        generator=None,
        text_encoder=None,
        vae=None,
    ):
        super().__init__()

        self.args = args

        self.generator = (
            WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
            if generator is None
            else generator
        )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        denoise_steps_cfg = getattr(args, "denoise_steps", None)
        legacy_steps_cfg = getattr(args, "denoising_step_list", None)
        if denoise_steps_cfg is None and legacy_steps_cfg is None:
            raise AttributeError(
                "Missing denoising steps config: expected `denoise_steps` (preferred) "
                "or legacy `denoising_step_list`."
            )
        if denoise_steps_cfg is not None and legacy_steps_cfg is not None and denoise_steps_cfg != legacy_steps_cfg:
            raise ValueError(
                "Both `denoise_steps` and legacy `denoising_step_list` are set but differ; "
                "please keep only one."
            )
        steps_cfg = denoise_steps_cfg if denoise_steps_cfg is not None else legacy_steps_cfg

        self.denoise_steps = self._normalize_denoising_steps(
            steps_cfg, warp=getattr(args, "warp_denoising_step", False)
        )
        # Backward-compatible alias (avoid breaking external code that still reads it).
        self.denoising_step_list = self.denoise_steps

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_clean = None
        self.crossattn_cache = None

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if hasattr(self.generator.model, "set_num_frame_per_block"):
            self.generator.model.set_num_frame_per_block(self.num_frame_per_block)
        else:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _normalize_denoising_steps(self, steps, warp: bool) -> torch.Tensor:
        step_tensor = torch.tensor(steps, dtype=torch.long)
        if not warp:
            return step_tensor

        timesteps = torch.cat(
            (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
        )
        return timesteps[1000 - step_tensor]

    def _reset_or_init_caches(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        if self.kv_cache_clean is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
            return

        for blk in range(self.num_transformer_blocks):
            self.crossattn_cache[blk]["is_init"] = False

        for blk in range(len(self.kv_cache_clean)):
            self.kv_cache_clean[blk]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device
            )
            self.kv_cache_clean[blk]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device
            )

    def _build_conditionals(
        self,
        *,
        text_prompts: List[str],
        noise: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cond = self.text_encoder(text_prompts=text_prompts)
        if audio_embeddings is not None:
            cond["audio_emb"] = audio_embeddings.to(device=noise.device, dtype=noise.dtype)
        if y is not None:
            cond["y"] = y.to(device=noise.device, dtype=noise.dtype)
        return cond

    def _decode_video(self, latents: torch.Tensor, profile: Optional[_ProfileHandles]) -> torch.Tensor:
        if profile is not None:
            profile.vae_start.record()

        video = self.vae.decode_to_pixel(latents, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile is not None:
            profile.vae_end.record()
            torch.cuda.synchronize()
        return video

    def _print_profile(self, profile: _ProfileHandles) -> None:
        diffusion_time = profile.diffusion_start.elapsed_time(profile.diffusion_end)
        init_time = profile.init_start.elapsed_time(profile.init_end)
        vae_time = profile.vae_start.elapsed_time(profile.vae_end)
        total_time = init_time + diffusion_time + vae_time

        print("Profiling results:")
        print(
            f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)"
        )
        print(
            f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)"
        )
        for i, t in enumerate(profile.block_times):
            print(
                f"    - Block {i} generation time: {t:.2f} ms ({100 * t / diffusion_time:.2f}% of diffusion)"
            )
        print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
        print(f"  - Total time: {total_time:.2f} ms")

    def _count_blocks(self, noise_frames: int, *, initial_latent: Optional[torch.Tensor]) -> int:
        if (not self.independent_first_frame) or (
            self.independent_first_frame and initial_latent is not None
        ):
            assert noise_frames % self.num_frame_per_block == 0
            return noise_frames // self.num_frame_per_block

        assert (noise_frames - 1) % self.num_frame_per_block == 0
        return (noise_frames - 1) // self.num_frame_per_block

    def _prefill_cache_for_rolling(
        self,
        *,
        initial_latent: Optional[torch.Tensor],
        output: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> int:
        prefix_frame = 1 if (self.independent_first_frame and initial_latent is not None) else 0
        if initial_latent is None:
            return prefix_frame

        num_input_frames = initial_latent.shape[1]
        zero_timestep = torch.zeros([batch_size, 1], device=device, dtype=torch.int64)

        start_frame = 0
        if self.independent_first_frame:
            assert (num_input_frames - 1) % self.num_frame_per_block == 0
            num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
            output[:, :1] = initial_latent[:, :1]
            w = slice_conditional_dict(conditional_dict, start_frame, start_frame + 1)
            self.generator(
                noisy_image_or_video=initial_latent[:, :1],
                conditional_dict=w,
                timestep=zero_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
            )
            start_frame += 1
        else:
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

        for _ in range(num_input_blocks):
            ref = initial_latent[:, start_frame : start_frame + self.num_frame_per_block]
            output[:, start_frame : start_frame + self.num_frame_per_block] = ref
            w = slice_conditional_dict(
                conditional_dict, start_frame, start_frame + self.num_frame_per_block
            )
            self.generator(
                noisy_image_or_video=ref,
                conditional_dict=w,
                timestep=zero_timestep * 0,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
            )
            start_frame += self.num_frame_per_block

        return prefix_frame

    def _prefill_cache_for_self_forcing(
        self,
        *,
        initial_latent: Optional[torch.Tensor],
        output: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> int:
        if initial_latent is None:
            return 0

        start_frame = 0
        zero_timestep = torch.zeros([batch_size, 1], device=device, dtype=torch.int64)
        output[:, :1] = initial_latent[:, :1]
        w = slice_conditional_dict(conditional_dict, start_frame, start_frame + 1)
        self.generator(
            noisy_image_or_video=initial_latent[:, :1],
            conditional_dict=w,
            timestep=zero_timestep * 0,
            kv_cache=self.kv_cache_clean,
            crossattn_cache=self.crossattn_cache,
            current_start=start_frame * self.frame_seq_length,
        )
        return 1

    def inference_avatar_forcing(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        progress: bool = False,
    ):
        """
        AvatarForcing inference (windowed forcing).

        Inputs:
            noise: (B, T_noise, C, H, W), frames to generate from noise.
            initial_latent: (B, T_prefix, C, H, W), optional prefix frames.
        Outputs:
            video: (B, T_total, 3, H, W) in [0, 1].
        """

        bsz, t_noise, c, h, w = noise.shape
        n_blocks = self._count_blocks(t_noise, initial_latent=initial_latent)
        t_prefix = 0 if initial_latent is None else initial_latent.shape[1]
        t_total = t_noise + t_prefix

        conditional_dict = self._build_conditionals(
            text_prompts=text_prompts, noise=noise, audio_embeddings=audio_embeddings, y=y
        )
        output = torch.zeros([bsz, t_total, c, h, w], device=noise.device, dtype=noise.dtype)

        prof = _make_profile(profile)
        if prof is not None:
            prof.init_start.record()

        self._reset_or_init_caches(batch_size=bsz, dtype=noise.dtype, device=noise.device)
        prefix_frame = self._prefill_cache_for_rolling(
            initial_latent=initial_latent,
            output=output,
            conditional_dict=conditional_dict,
            batch_size=bsz,
            device=noise.device,
        )

        if prof is not None:
            prof.init_end.record()
            torch.cuda.synchronize()
            prof.diffusion_start.record()

        # window plan (in blocks)
        window_blocks = len(self.denoise_steps)
        window_num = n_blocks + window_blocks - 1
        start_blocks = [
            max(0, wi - window_blocks + 1) for wi in range(window_num)
        ]
        end_blocks = [
            min(n_blocks - 1, wi) for wi in range(window_num)
        ]

        noisy_cache = torch.zeros([bsz, t_total, c, h, w], device=noise.device, dtype=noise.dtype)

        expanded = (
            self.denoise_steps.to(device=noise.device, dtype=torch.float32)
            .flip(0)
            .repeat_interleave(self.num_frame_per_block)
        )
        shared_timestep = expanded.unsqueeze(0).repeat(bsz, 1)

        window_iter = range(window_num)
        if progress and tqdm is not None:
            window_iter = tqdm(
                window_iter,
                desc="AvatarForcing windows",
                dynamic_ncols=True,
                leave=False,
                mininterval=1.0,
            )

        for wi in window_iter:
            if prof is not None:
                prof.block_start.record()

            sb, eb = start_blocks[wi], end_blocks[wi]

            frame_s = prefix_frame + sb * self.num_frame_per_block
            frame_e = prefix_frame + (eb + 1) * self.num_frame_per_block
            cur_frames = frame_e - frame_s

            noise_s = frame_s - prefix_frame
            noise_e = frame_e - prefix_frame

            # noisy_input: mix last block noise with cached frames (or only cache at tail)
            if cur_frames == window_blocks * self.num_frame_per_block or frame_s == prefix_frame:
                noisy_input = torch.cat(
                    [
                        noisy_cache[:, frame_s : frame_e - self.num_frame_per_block],
                        noise[:, noise_e - self.num_frame_per_block : noise_e],
                    ],
                    dim=1,
                )
            else:
                noisy_input = noisy_cache[:, frame_s:frame_e]

            if cur_frames == window_blocks * self.num_frame_per_block:
                cur_timestep = shared_timestep
            elif frame_s == prefix_frame:
                cur_timestep = shared_timestep[:, -cur_frames:]
            elif noise_e == t_noise:
                cur_timestep = shared_timestep[:, :cur_frames]
            else:
                raise ValueError(
                    "Unexpected window layout; expected full, first or last window."
                )

            cond_window = slice_conditional_dict(conditional_dict, frame_s, frame_e)
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_window,
                timestep=cur_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=frame_s * self.frame_seq_length,
            )
            output[:, frame_s:frame_e] = denoised_pred

            with torch.no_grad():
                for blk in range(sb, eb + 1):
                    blk_slice = cur_timestep[
                        :,
                        (blk - sb) * self.num_frame_per_block : (blk - sb + 1)
                        * self.num_frame_per_block,
                    ]
                    blk_ts = blk_slice.mean().item()
                    matches = torch.abs(self.denoise_steps - blk_ts) < 1e-4
                    idx = torch.nonzero(matches, as_tuple=True)[0]
                    if idx == len(self.denoise_steps) - 1:
                        continue

                    next_ts = self.denoise_steps[idx + 1].to(noise.device)

                    out_s = prefix_frame + blk * self.num_frame_per_block
                    out_e = out_s + self.num_frame_per_block

                    noisy_cache[:, out_s:out_e] = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_ts
                        * torch.ones(
                            [bsz * cur_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])[
                        :,
                        (blk - sb) * self.num_frame_per_block : (blk - sb + 1)
                        * self.num_frame_per_block,
                    ]

            with torch.no_grad():
                cache_timestep = torch.ones_like(cur_timestep) * self.args.context_noise
                cache_latents = denoised_pred[:, : self.num_frame_per_block]
                cache_timestep = cache_timestep[:, : self.num_frame_per_block]
                cache_cond = slice_conditional_dict(
                    conditional_dict, frame_s, frame_s + self.num_frame_per_block
                )
                self.generator(
                    noisy_image_or_video=cache_latents,
                    conditional_dict=cache_cond,
                    timestep=cache_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=frame_s * self.frame_seq_length,
                    updating_cache=True,
                )

            if prof is not None:
                prof.block_end.record()
                torch.cuda.synchronize()
                prof.block_times.append(prof.block_start.elapsed_time(prof.block_end))

        if prof is not None:
            prof.diffusion_end.record()
            torch.cuda.synchronize()

        video = self._decode_video(output, prof)
        if prof is not None:
            self._print_profile(prof)

        return (video, output) if return_latents else video

    def inference_self_forcing(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
    ):
        """
        AvatarForcing self-forcing inference.

        Inputs:
            noise: (B, T_noise, C, H, W), frames to generate from noise.
            initial_latent: (B, T_prefix, C, H, W), optional prefix frames.
        Outputs:
            video: (B, T_total, 3, H, W) in [0, 1].
        """

        bsz, t_noise, c, h, w = noise.shape
        n_blocks = self._count_blocks(t_noise, initial_latent=initial_latent)
        t_prefix = 0 if initial_latent is None else initial_latent.shape[1]
        t_total = t_noise + t_prefix

        conditional_dict = self._build_conditionals(
            text_prompts=text_prompts, noise=noise, audio_embeddings=audio_embeddings, y=y
        )
        output = torch.zeros([bsz, t_total, c, h, w], device=noise.device, dtype=noise.dtype)

        prof = _make_profile(profile)
        if prof is not None:
            prof.init_start.record()

        self._reset_or_init_caches(batch_size=bsz, dtype=noise.dtype, device=noise.device)
        start_frame = self._prefill_cache_for_self_forcing(
            initial_latent=initial_latent,
            output=output,
            conditional_dict=conditional_dict,
            batch_size=bsz,
            device=noise.device,
        )

        if prof is not None:
            prof.init_end.record()
            torch.cuda.synchronize()
            prof.diffusion_start.record()

        num_input_frames = 0 if initial_latent is None else initial_latent.shape[1]

        block_sizes = [self.num_frame_per_block] * n_blocks
        if self.independent_first_frame and initial_latent is None:
            block_sizes = [1] + block_sizes

        num_steps = len(self.denoise_steps)
        for blk_idx, cur_frames in enumerate(block_sizes):
            if prof is not None:
                prof.block_start.record()

            noisy_input = noise[
                :,
                start_frame - num_input_frames : start_frame + cur_frames - num_input_frames,
            ]
            cond_window = slice_conditional_dict(conditional_dict, start_frame, start_frame + cur_frames)

            for si, cur_step in enumerate(self.denoise_steps):
                timestep = torch.ones(
                    [bsz, cur_frames], device=noise.device, dtype=torch.int64
                ) * cur_step
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=cond_window,
                    timestep=timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * self.frame_seq_length,
                )

                if si < num_steps - 1:
                    next_step = self.denoise_steps[si + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_step
                        * torch.ones(
                            [bsz * cur_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])

            output[:, start_frame : start_frame + cur_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(),
            ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_window,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * self.frame_seq_length,
                    updating_cache=True,
                )

            start_frame += cur_frames

            if prof is not None:
                prof.block_end.record()
                torch.cuda.synchronize()
                prof.block_times.append(prof.block_start.elapsed_time(prof.block_end))

        if prof is not None:
            prof.diffusion_end.record()
            torch.cuda.synchronize()

        video = self._decode_video(output, prof)
        if prof is not None:
            self._print_profile(prof)

        return (video, output) if return_latents else video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache_clean = []
        kv_cache_size = 1560 * 7

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append(
                {
                    "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache_clean = kv_cache_clean

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache


class CausalInferencePipeline(AvatarForcingInferencePipeline):
    pass
