"""冻结的 CLIP，给全库商品编码。直接从 HF 加载。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP = "openai/clip-vit-base-patch32"
MAX_TEXT_LEN = 500


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEncoder:
    def __init__(self, model_name: str = DEFAULT_CLIP, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.n_truncated = 0

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim

    def meta(self) -> dict:
        return {
            "model": self.model_name,
            "hf_commit": getattr(self.model.config, "_commit_hash", None),
            "dim": self.dim,
            "max_text_len": MAX_TEXT_LEN,
            "l2_normalized": True,
            "dtype": "float32",
            "n_text_truncated": self.n_truncated,
        }

    @torch.no_grad()
    def encode_text(self, titles: list[str], batch_size: int = 256) -> np.ndarray:
        """返回 (n, dim) float32，每行已 L2 归一化。超过 77 token 的标题会被截断并计数。"""
        tok = self.processor.tokenizer
        self.n_truncated += sum(len(ids) > MAX_TEXT_LEN for ids in tok(titles)["input_ids"])
        out = []
        for i in tqdm(range(0, len(titles), batch_size), desc="clip text"):
            batch = tok(titles[i:i + batch_size], padding=True, truncation=True,
                        max_length=MAX_TEXT_LEN, return_tensors="pt").to(self.device)
            out.append(self._norm(self.model.get_text_features(**batch)))
        return np.concatenate(out)

    @torch.no_grad()
    def encode_images(self, paths: list[Path], batch_size: int = 128) -> np.ndarray:
        """返回 (n, dim) float32，每行已 L2 归一化。图片统一转 RGB。"""
        out = []
        for i in tqdm(range(0, len(paths), batch_size), desc="clip image"):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            batch = self.processor(images=imgs, return_tensors="pt").to(self.device)
            out.append(self._norm(self.model.get_image_features(**batch)))
        return np.concatenate(out)

    @staticmethod
    def _norm(x: torch.Tensor) -> np.ndarray:
        x = x.float()
        return torch.nn.functional.normalize(x, dim=-1).cpu().numpy().astype(np.float32)
