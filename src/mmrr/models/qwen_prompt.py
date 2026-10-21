"""Pointwise 重排的提示词模板（我们自己定，zero-shot 推理用）。

参考：
- 快手 OpenOneRec `data/onerec_data/sft/label_pred.py`：system 角色 + 历史区块 + 针对候选的是/否判断题
- MiniOneRec `data.py`：Alpaca 式 "### Instruction / ### User Input / ### Response"，
  历史写成 "The user has ... in chronological order"，标题加引号
两者都是给"预测下一个 SID"训练用的，我们把物品表示从 SID 换成 标题(+图片)，任务换成 Yes/No。

约定：
- 推理时模板固定（不像训练时随机抽句式），保证结果可复现
- 每个物品"标题在前、图片在后"：vLLM 前缀缓存按 token 哈希，图片占位 token 都一样，
  标题放前面才能在图片之前就分叉，避免错误复用别的图片的 KV
- 改任何模板文字都要升该模板的版本号，meta.json 会记录 "<name>@<version>"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from mmrr.data.schema import Item
from mmrr.paths import resolve_data_path

ANSWER_YES, ANSWER_NO = "Yes", "No"

# 数据集 -> 提示词里的物品名词
CATEGORY = {"microlens": "video", "movielens": "movie", "netflix": "movie"}


@dataclass(frozen=True)
class PromptConfig:
    template: str = "onerec"
    history_images: bool = False   # True: 历史物品也放图（约 6 倍图片 token）
    candidate_image: bool = True
    # Qwen2.5-VL 每 28x28 像素一个图片 token，这里设上下限控制成本
    min_pixels: int = 64 * 28 * 28
    max_pixels: int = 256 * 28 * 28

    def to_dict(self) -> dict:
        return {"prompt": template_id(self.template), **asdict(self)}


class _Content:
    """按顺序拼 text / image 片段，相邻文本自动合并。"""

    def __init__(self, cfg: PromptConfig):
        self.cfg, self.parts = cfg, []

    def text(self, s: str) -> "_Content":
        if self.parts and self.parts[-1]["type"] == "text":
            self.parts[-1]["text"] += s
        else:
            self.parts.append({"type": "text", "text": s})
        return self

    def image(self, item: Item, on: bool) -> "_Content":
        if on and item.image_path is not None:  # 缺图的物品只给标题
            self.parts.append({"type": "image", "image": resolve_data_path(item.image_path).as_posix(),
                               "min_pixels": self.cfg.min_pixels, "max_pixels": self.cfg.max_pixels})
        return self


def _q(title: str) -> str:
    return '"' + title.replace('"', "'") + '"'


# ---------------- 模板 1：OpenOneRec 风格（默认） ----------------
def _onerec(history: list[Item], cand: Item, cfg: PromptConfig, cat: str) -> list[dict]:
    system = (f"You are a personalized recommendation system. Based on a user's interaction history, "
              f"you predict whether the user will be interested in a new {cat}.")
    c = _Content(cfg).text(f"The user has watched the following {cat}s in chronological order "
                           f"(oldest first):\n")
    for i, it in enumerate(history, 1):
        c.text(f"{i}. {_q(it.title)}\n").image(it, cfg.history_images)
    c.text(f"\nCandidate {cat}: {_q(cand.title)}\n").image(cand, cfg.candidate_image)
    c.text(f"\nWill the user watch the candidate {cat} next? Answer {ANSWER_YES} or {ANSWER_NO}.")
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": c.parts}]


# ---------------- 模板 2：MiniOneRec 风格（Alpaca 式） ----------------
def _minionerec(history: list[Item], cand: Item, cfg: PromptConfig, cat: str) -> list[dict]:
    c = _Content(cfg).text(
        "Below is an instruction that describes a task, paired with an input that provides further "
        "context. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\nGiven the user's interaction history, will the user watch the candidate "
        f"{cat} next? Answer {ANSWER_YES} or {ANSWER_NO}.\n\n"
        f"### User Input:\nThe user has interacted with the following {cat}s in chronological order: ")
    for i, it in enumerate(history):
        c.text(("" if i == 0 else ",\t") + _q(it.title)).image(it, cfg.history_images)
    c.text(f"\nCandidate {cat}: {_q(cand.title)}\n").image(cand, cfg.candidate_image)
    c.text("\n### Response:")
    return [{"role": "user", "content": c.parts}]


TEMPLATES: dict[str, tuple[str, Callable]] = {
    "onerec": ("v1", _onerec),
    "minionerec": ("v1", _minionerec),
}


def template_id(name: str) -> str:
    return f"{name}@{TEMPLATES[name][0]}"


def build_messages(history: list[Item], candidate: Item, cfg: PromptConfig,
                   dataset: str = "microlens") -> list[dict]:
    if cfg.template not in TEMPLATES:
        raise KeyError(f"未知模板 {cfg.template!r}，可选 {list(TEMPLATES)}")
    return TEMPLATES[cfg.template][1](history, candidate, cfg, CATEGORY[dataset])


def max_images_per_prompt(max_history: int, cfg: PromptConfig) -> int:
    return max_history * cfg.history_images + cfg.candidate_image


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """答案位置的 Yes/No 必须各是 1 个 token，否则直接报错（不静默算错）。"""
    ids = []
    for w in (ANSWER_YES, ANSWER_NO):
        t = tokenizer.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{w!r} 被切成 {len(t)} 个 token: {t}，需要改成整段答案 likelihood")
        ids.append(t[0])
    return ids[0], ids[1]
