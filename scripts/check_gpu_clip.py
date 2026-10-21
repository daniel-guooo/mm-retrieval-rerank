"""Small GPU smoke test of the actual project CLIP encoder; no full catalog run."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmrr.data.loaders import clean_title, image_files, read_titles
from mmrr.models.clip_encoder import ClipEncoder


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Local CLIP model directory")
    ap.add_argument("--dataset", default="microlens")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item()
    del x, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("GPU calculation OK", flush=True)
    titles = read_titles(args.dataset)
    images = image_files(args.dataset)
    ids = sorted(titles.keys() & images.keys())[:16]
    assert len(ids) == 16, "Need 16 valid paired items for this smoke test"
    start = time.perf_counter()
    encoder = ClipEncoder(args.model, device="cuda")
    assert next(encoder.model.parameters()).is_cuda
    assert encoder.meta()["max_text_len"] == encoder.model.config.text_config.max_position_embeddings
    text = encoder.encode_text([clean_title(titles[i]) for i in ids], batch_size=8)
    image = encoder.encode_images([images[i] for i in ids], batch_size=4)
    before = encoder.n_truncated
    long_text = encoder.encode_text(["a basketball video " * 150], batch_size=1)
    assert encoder.n_truncated == before + 1, "Long-title truncation not counted"
    for features, shape in ((text, (16, 512)), (image, (16, 512)), (long_text, (1, 512))):
        assert features.shape == shape
        assert features.dtype == np.float32 and np.isfinite(features).all()
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
    torch.cuda.synchronize()
    report = {
        "status": "PASS", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in ("torchvision", "transformers", "accelerate", "qwen-vl-utils")},
        "dataset": args.dataset, "item_ids": ids,
        "text_shape": list(text.shape), "image_shape": list(image.shape),
        "finite_normalized_features": True, "long_title_truncation": "PASS",
        "max_text_length": encoder.meta()["max_text_len"],
        "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_torch_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "seconds_including_model_load": round(time.perf_counter() - start, 2),
        "scope": "16 paired items, one synthetic long title; no Qwen or full-catalog benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
