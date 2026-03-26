import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from tqdm import tqdm
class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("./wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name="./wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="./wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            # use_temporal_cache=True,
            model_init_kwargs=None,
    ):
        super().__init__()

        if is_causal:
            model_class = CausalWanModel
        else:
            model_class = WanModel
        init_kwargs = {} if model_init_kwargs is None else dict(model_init_kwargs)
        self.model = model_class(**init_kwargs)
        self.model.eval()

        # if is_causal and hasattr(self.model, "set_use_temporal_cache"): ### for no temporal anchor alation study
        #     self.model.set_use_temporal_cache(use_temporal_cache)

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # Token sequence length for default latent shape (F=21, H=60, W=104) with 2x2 spatial patches:
        # 21 * (60/2) * (104/2) = 32760
        self.seq_len = 32760
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Expects features with last dim = atten_dim * 3 + time_embed_dim
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        updating_cache: Optional[bool] = False
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        audio_emb = conditional_dict["audio_emb"]
        y = conditional_dict["y"]
        # assert audio_emb.shape[1] == (noisy_image_or_video.shape[1] - 1) // 4 + 1

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                y=y,
                audio_emb=audio_emb,
                cache_start=cache_start,
                updating_cache=updating_cache
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    y=y,
                    audio_emb=audio_emb,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        y=y,
                        audio_emb=audio_emb,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        y=y,
                        audio_emb=audio_emb,
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    @torch.no_grad()
    def log_video(
        self,
        lat,
        prompt_emb_posi,
        prompt_emb_nega,
        fixed_frame=0, # lat frames
        image_emb={},
        audio_emb=0.0,
        cfg_scale=5.0,
        audio_cfg_scale=5.0,
        num_inference_steps=50,
        denoising_strength=1.0,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        return_latent=False,
    ):

        '''
            Test for Omnihuman
            latents = self.model.real_score.log_video(
                img_lat,
                conditional_dict['prompt_embeds'],
                unconditional_dict['prompt_embeds'],
                1,
                {"y": image_emb},
                {"audio_emb": audio_emb},
                num_inference_steps=50,
                cfg_scale=4.5,
                audio_cfg_scale=4.5,
                return_latent=True)

            frames = self.model.vae.decode_to_pixel(latents_fake.permute(0, 2, 1, 3, 4).to(torch.bfloat16))
            frames = (frames.permute(0, 2, 1, 3, 4).float() + 1) / 2
            image = (frames.clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
        '''
        ode_latents = []  # list of (B,C,F,H,W)

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        lat = torch.cat([lat, torch.zeros_like(lat[:, :, :1].repeat(1, 1, 20, 1, 1))], dim=2)
        latents = lat.clone()
        latents = torch.randn_like(latents)

        # Encode prompts
        # self.load_models_to_device(["text_encoder"])
        # prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        # if cfg_scale != 1.0:
        #     prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = {}
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, disable=False and torch.distributed.get_rank() != 0)):
            if fixed_frame > 0: # new
                latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]
            ode_latents.append(latents.detach().clone())
            timestep = timestep.unsqueeze(0).to(dtype=torch.bfloat16, device=latents.device)

            # Inference
            noise_pred_posi = self.model(latents, timestep, prompt_emb_posi, **image_emb, **audio_emb, **extra_input)

            if cfg_scale != 1.0:
                audio_emb_uc = {}
                for key in audio_emb.keys():
                    audio_emb_uc[key] = torch.zeros_like(audio_emb[key])
                if audio_cfg_scale == cfg_scale:
                    noise_pred_nega = self.model(latents, timestep, prompt_emb_nega, **image_emb, **audio_emb_uc, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            # Scheduler
            t =  self.scheduler.timesteps[progress_id].unsqueeze(0).to(dtype=torch.bfloat16, device=latents.device)
            latents = self.scheduler.step(noise_pred, t, latents)

        if fixed_frame > 0: # new
            latents[:, :, :fixed_frame] = lat[:, :, :fixed_frame]
        ode_latents.append(latents.detach().clone())

        # ode_latents: list length = num_inference_steps + 1, each (B,C,F,H,W)
        ode_latent = torch.stack(ode_latents, dim=1)  # (B, S, C, F, H, W)
        ode_latent = ode_latent.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B,S,F,C,H,W)

        # if return_latent:
        #     return latents
        return ode_latent
