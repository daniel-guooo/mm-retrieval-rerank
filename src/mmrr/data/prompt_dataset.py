"""Lazy user/candidate prompt adapter, reconstructed from the tracked callers.

Targets are evaluation labels only; build_messages never receives a target.
"""
from bisect import bisect_right
from dataclasses import dataclass

from mmrr.models.qwen_prompt import PromptConfig, build_messages, max_images_per_prompt


def render_text(messages):
    return "\n".join(part["text"] for message in messages for part in message["content"] if part["type"] == "text")


@dataclass(frozen=True)
class PromptExample:
    user: str
    candidate: int
    position: int
    label: int
    messages: list
    image_paths: list[str]


class RerankPromptDataset:
    def __init__(self, records, catalog, cfg=None, dataset="microlens"):
        self.records = list(records)
        self.catalog, self.cfg, self.dataset = catalog, cfg or PromptConfig(), dataset
        self.n_users = len(self.records)
        self.offsets = [0]
        for record in self.records:
            for item_id in [*record["history"], *record["candidates"]]:
                if item_id not in catalog:
                    raise KeyError(item_id)
            self.offsets.append(self.offsets[-1] + len(record["candidates"]))
        self.max_images_per_prompt = max_images_per_prompt(max((len(r["history"]) for r in self.records), default=0), self.cfg)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = bisect_right(self.offsets, index) - 1
        record = self.records[row]
        position = index - self.offsets[row]
        candidate = record["candidates"][position]
        messages = build_messages([self.catalog[i] for i in record["history"]], self.catalog[candidate], self.cfg, self.dataset)
        images = [part["image"] for message in messages for part in message["content"] if part["type"] == "image"]
        return PromptExample(record["user"], candidate, position, int(candidate == record["target"]), messages, images)

    def iter_user_batches(self, users_per_call):
        if users_per_call < 1:
            raise ValueError("users_per_call must be positive")
        for start in range(0, self.n_users, users_per_call):
            end = min(start + users_per_call, self.n_users)
            yield self.records[start:end], [self[i] for i in range(self.offsets[start], self.offsets[end])]

    def meta(self):
        return {**self.cfg.to_dict(), "dataset": self.dataset, "n_users": self.n_users,
                "n_prompts": len(self), "max_images_per_prompt": self.max_images_per_prompt}
