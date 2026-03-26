import argparse
import torch
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    AvatarForcingInferencePipeline
)
from utils.dataset import TextImageAudioPairDataset
from utils.misc import set_seed
from utils.inject import _apply_lora
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import math
import subprocess

class ResizeKeepRatioArea16:
    def __init__(self, area_hw=(480, 832), div=16):
        self.A, self.d = area_hw[0] * area_hw[1], div

    def __call__(self, img):
        w, h = img.size
        s = min(1.0, math.sqrt(self.A / (h * w)))
        nh = max(self.d, int(h * s) // self.d * self.d)
        nw = max(self.d, int(w * s) // self.d * self.d)
        return F.resize(img, (nh, nw), interpolation=InterpolationMode.BILINEAR, antialias=True)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the I2V prompt file")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of frames to generate (including the first frame for I2V)")
parser.add_argument("--i2v", action="store_true", help="Enable image-to-video inference (required by this script)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

def _collate_skip_none(batch):
    from torch.utils.data._utils.collate import default_collate

    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

pipeline = AvatarForcingInferencePipeline(config, device=device)
if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if args.use_ema:
        state_dict_to_load = state_dict['generator_ema']
        def remove_fsdp_prefix(state_dict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if "_fsdp_wrapped_module." in key:
                    new_key = key.replace("_fsdp_wrapped_module.", "")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        state_dict_to_load = remove_fsdp_prefix(state_dict_to_load)
    else:
        state_dict_to_load = state_dict['generator']
    pipeline.generator.model = _apply_lora(pipeline.generator.model, config["models"]["lora"])
    pipeline.generator.load_state_dict(state_dict_to_load)

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

# Create dataset
data_cfg = dict(config.data)             
if args.data_path is not None:         
    data_cfg["path"] = args.data_path
if args.num_output_frames is not None:
    data_cfg["teacher_len"] = args.num_output_frames*4 + 80
if not args.i2v:
    raise ValueError("This script currently supports I2V only; pass `--i2v`.")

assert not dist.is_initialized(), "I2V does not support distributed inference yet"
transform = transforms.Compose([
    ResizeKeepRatioArea16((480, 832), 16),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset = TextImageAudioPairDataset(**data_cfg, transform=transform)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    sampler=sampler,
    num_workers=0,
    drop_last=False,
    collate_fn=_collate_skip_none,
)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    if batch_data is None:
        continue
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    prompt = batch['prompts'][0]
    prompts = [prompt] * args.num_samples

    image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(
        device=device, dtype=torch.bfloat16
    )

    initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
    initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

    img_lat = initial_latent.permute(0, 2, 1, 3, 4)
    msk = torch.zeros_like(img_lat.repeat(1, 1, args.num_output_frames + 20, 1, 1)[:, :1])
    image_cat = img_lat.repeat(1, 1, args.num_output_frames + 20, 1, 1)
    msk[:, :, 1:] = 1
    y = torch.cat([image_cat, msk], dim=1)

    h, w = initial_latent.shape[-2], initial_latent.shape[-1]
    sampled_noise = torch.randn(
        (args.num_samples, args.num_output_frames - 1, 16, h, w),
        device=initial_latent.device,
        dtype=initial_latent.dtype,
    )

    # Generate video frames
    video = pipeline.inference_avatar_forcing(
        noise=sampled_noise,
        text_prompts=prompts,
        audio_embeddings=batch['audio_emb'],
        y=y,
        return_latents=False,
        initial_latent=initial_latent,
    )

    video = 255.0 * rearrange(video, 'b t c h w -> b t h w c').cpu()

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=25)
            subprocess.run(
                f"ffmpeg -y -i \"{output_path}\" -i \"{batch_data['wav_path'][0]}\" "
                f"-c:v copy -c:a aac -shortest \"{output_path}.tmp.mp4\" && mv \"{output_path}.tmp.mp4\" \"{output_path}\"",
                shell=True, check=True
            )
