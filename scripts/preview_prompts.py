"""打印几条真实的 prompt，检查模板长什么样（不需要模型、不需要 GPU）。

    python scripts/preview_prompts.py --split val --n 2 --template onerec
"""
import argparse

from mmrr.data.loaders import load_catalog
from mmrr.data.prompt_dataset import RerankPromptDataset, render_text
from mmrr.io import read_jsonl
from mmrr.models.qwen_prompt import TEMPLATES, PromptConfig
from mmrr.paths import candidates_path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="microlens")
ap.add_argument("--split", default="val")
ap.add_argument("--template", default="onerec", choices=list(TEMPLATES))
ap.add_argument("--history-images", action="store_true")
ap.add_argument("--n", type=int, default=2)
args = ap.parse_args()

recs = []
for r in read_jsonl(candidates_path(args.dataset, args.split)):
    recs.append(r)
    if len(recs) >= 1:
        break
ds = RerankPromptDataset(recs, load_catalog(args.dataset),
                         PromptConfig(template=args.template, history_images=args.history_images), args.dataset)
for i in range(min(args.n, len(ds))):
    ex = ds[i]
    print(f"===== user={ex.user} candidate={ex.candidate} pos={ex.position} label={ex.label} =====")
    print(render_text(ex.messages), "\n")
