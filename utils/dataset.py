from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import pandas as pd
import math
import cv2
from decord import VideoReader
import csv
import shlex
from transformers import Wav2Vec2FeatureExtractor

class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.data_path = data_path
        self.max_pair = max_pair

        with open(data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"audio_emb", "y", "ode_latents", "caption"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"CSV must contain columns {sorted(required)}, got {reader.fieldnames}")
            self.rows = list(reader)

        if len(self.rows) == 0:
            raise ValueError(f"Empty CSV: {data_path}")

    def __len__(self):
        return min(len(self.rows), self.max_pair)

    def __getitem__(self, idx):
        row = self.rows[idx]

        ode_latent = torch.load(row["ode_latents"], map_location="cpu", weights_only=True).to(torch.bfloat16)
        ode_latent = ode_latent.squeeze(0)

        audio_emb = torch.load(row["audio_emb"], map_location="cpu", weights_only=True).to(torch.bfloat16)
        audio_emb = audio_emb.squeeze(0)
        y = torch.load(row["y"], map_location="cpu", weights_only=True).to(torch.bfloat16)
        y = y.squeeze(0)
        prompts = row["caption"]
        return {
            "prompts": prompts,
            "ode_latent": ode_latent,
            "audio_emb": audio_emb,
            "y": y,
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data

class TextVideoAudioPairDataset(Dataset):
    def __init__(self,
                 path,
                 wav2vec_path=None,
                 target_area=(832,480),
                 divisor=64,
                 teacher_len=20,
                 fps=16,
                 max_samples=None):

        self.csv_path = path
        self.wav2vec_path = wav2vec_path
        self.divisor = divisor
        self.target_area = tuple(target_area)
        self.teacher_len = teacher_len
        self.fps = fps

        # Load CSV data
        print(f"Loading S2V dataset from {self.csv_path}")
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Loaded {len(self.data)} samples from CSV")
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file {self.csv_path}: {e}")

        # Filter samples
        if max_samples is not None and max_samples > 0:
            self.data = self.data.head(max_samples)
            print(f"Limited to {len(self.data)} samples")

        # Validate required columns
        required_columns = ['video_path_caption', 'video_path', 'audio_path']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")

        # Filter out samples with missing essential data
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['video_path_caption', 'video_path', 'audio_path'])
        filtered_count = len(self.data)
        if filtered_count < initial_count:
            print(f"Filtered {initial_count - filtered_count} samples with missing data")
        print(f"Final dataset size: {len(self.data)} samples")

        #==================== load audio ========================
        from wan.models.wav2vec import Wav2VecModel
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.wav2vec_path)
        self.audio_encoder = Wav2VecModel.from_pretrained(self.wav2vec_path, local_files_only=True)
        self.audio_encoder.feature_extractor._freeze_parameters()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = row['video_path']

        vr = VideoReader(video_path)
        frame = vr[0].asnumpy()
        frame = cv2.resize(frame, (832, 480), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        tensor = tensor.unsqueeze(1)

        sample = {
            'prompts': ' '.join(row['video_path_caption'].split()[7:]),
            'ref_pixel_values': tensor,
            'idx': idx,
        }

        audio_path = row.get('audio_path')
        sample['video_path'] = str(video_path) if pd.notna(video_path) and os.path.exists(video_path) else None

        audio = np.load(audio_path)
        input_values = np.squeeze(
            self.wav_feature_extractor(audio, sampling_rate=16000).input_values
        )
        input_values = torch.from_numpy(input_values).float().unsqueeze(0)

        total_len = math.ceil(len(input_values[0]) / 16000 * self.fps)
        if total_len < self.teacher_len:
            # print(f"[Skip] audio too short: {audio_path}")
            return None  #

        max_audio_len = self.teacher_len * int(16000 / self.fps)
        input_values = input_values[:, :max_audio_len]

        with torch.no_grad():
            hidden_states = self.audio_encoder(
                input_values, seq_len=self.teacher_len, output_hidden_states=True
            )
            audio_embeddings = hidden_states.last_hidden_state
            for mid_hidden_states in hidden_states.hidden_states:
                audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)

        audio_embeddings = audio_embeddings.squeeze(0)
        audio_prefix = torch.zeros_like(audio_embeddings[:1])
        sample['audio_emb'] = torch.cat([audio_prefix, audio_embeddings], dim=0)

        return sample

class TextImageAudioPairDataset(Dataset):
    def __init__(
        self,
        path,
        wav2vec_path,
        target_area=(832, 480),
        teacher_len=20,
        transform=None,
        fps=16,
        max_samples=None,
    ):
        self.txt_path = path
        self.target_area = tuple(target_area)
        self.teacher_len = teacher_len
        self.fps = fps
        self.transform = transform

        with open(self.txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if max_samples is not None and max_samples > 0:
            lines = lines[:max_samples]

        self.data = []
        for ln in lines:
            parts = shlex.split(ln)
            if len(parts) != 3:
                continue
            img, wav, cap = parts
            self.data.append({
                "image_path": img,
                "audio_path": wav,
                "caption": cap,
            })

        # ---- audio encoder ----
        from wan.models.wav2vec import Wav2VecModel
        self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_path, local_files_only=True)
        self.audio_encoder.feature_extractor._freeze_parameters()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = row["image_path"]
        image = Image.open(img_path).convert('RGB')
        # Apply transform if specified
        if self.transform:
            img = self.transform(image)

        # if not os.path.exists(img_path):
        #     return None
        # frame = cv2.imread(img_path)
        # if frame is None:
        #     return None
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, self.target_area, interpolation=cv2.INTER_AREA)
        # img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        # img = img * 2.0 - 1.0
        # img = img.unsqueeze(1)

        sample = {
            "prompts": row["caption"],
            "image": img,
            "idx": idx,
        }

        # ---- audio (wav @16k) ----
        wav_path = row["audio_path"]
        if not os.path.exists(wav_path):
            return None

        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import torchaudio
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio).unsqueeze(0), sr, 16000
            ).squeeze(0).numpy()

        input_values = np.squeeze(
            self.wav_feature_extractor(audio, sampling_rate=16000).input_values
        )
        input_values = torch.from_numpy(input_values).float().unsqueeze(0)

        total_len = math.ceil(len(input_values[0]) / 16000 * self.fps)
        if total_len < self.teacher_len:
            return None

        max_audio_len = self.teacher_len * int(16000 / self.fps)
        input_values = input_values[:, :max_audio_len]

        with torch.no_grad():
            hs = self.audio_encoder(
                input_values, seq_len=self.teacher_len, output_hidden_states=True
            )
            audio_emb = hs.last_hidden_state
            for h in hs.hidden_states:
                audio_emb = torch.cat([audio_emb, h], dim=-1)

        audio_emb = audio_emb.squeeze(0)
        sample["audio_emb"] = torch.cat(
            [torch.zeros_like(audio_emb[:1]), audio_emb], dim=0
        )
        sample['wav_path'] = wav_path
        return sample
