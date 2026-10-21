"""Qwen2.5-VL zero-shot pointwise 打分：s = log P(Yes) − log P(No)。

打分器只吃 messages（由 mmrr.data.prompt_dataset 拼好），和模板无关：
    scorer.score_messages(list_of_messages) -> np.ndarray (n,)
- QwenScorerHF：纯 transformers，作参考实现 / 小样本对数
- QwenScorerVLLM：正式跑（Linux/WSL2 + CUDA，如 4070），开前缀缓存

不做采样生成，只取答案位置下一个 token 的分布，结果确定。
"""
from __future__ import annotations

import numpy as np

from mmrr.models.qwen_prompt import yes_no_token_ids

DEFAULT_QWEN = "Qwen/Qwen2.5-VL-3B-Instruct"
Messages = list[dict]


class _Base:
    model_name: str
    yes_id: int
    no_id: int
    backend: str

    def score_messages(self, msgs: list[Messages]) -> np.ndarray:
        raise NotImplementedError

    def meta(self) -> dict:
        return {"backend": self.backend, "model": self.model_name,
                "yes_token_id": self.yes_id, "no_token_id": self.no_id,
                "score": "logP(Yes) - logP(No) at first answer token"}


class QwenScorerHF(_Base):
    backend = "hf"

    def __init__(self, model_name: str = DEFAULT_QWEN, device: str | None = None, batch_size: int = 8):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        from mmrr.models.clip_encoder import pick_device

        self.model_name, self.batch_size = model_name, batch_size
        self.device = device or pick_device()
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(self.device, torch.float32)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"  # 左 padding，最后一个位置就是答案位置
        self.yes_id, self.no_id = yes_no_token_ids(self.processor.tokenizer)
        self.hf_commit = getattr(self.model.config, "_commit_hash", None)

    def meta(self) -> dict:
        return {**super().meta(), "hf_commit": self.hf_commit, "batch_size": self.batch_size}

    def score_messages(self, msgs: list[Messages]) -> np.ndarray:
        import torch
        from qwen_vl_utils import process_vision_info

        out = []
        with torch.no_grad():
            for i in range(0, len(msgs), self.batch_size):
                chunk = msgs[i:i + self.batch_size]
                texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                         for m in chunk]
                images, _ = process_vision_info(chunk)
                inputs = self.processor(text=texts, images=images, padding=True,
                                        return_tensors="pt").to(self.device)
                logp = self.model(**inputs).logits[:, -1, :].float().log_softmax(-1)
                out.append((logp[:, self.yes_id] - logp[:, self.no_id]).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.empty(0, np.float32)


class QwenScorerVLLM(_Base):
    """4070 (12GB) 的默认参数：bf16 权重约 7.5GB，剩下给 KV cache。"""
    backend = "vllm"

    def __init__(self, model_name: str = DEFAULT_QWEN, max_images_per_prompt: int = 1,
                 min_pixels: int = 64 * 28 * 28, max_pixels: int = 256 * 28 * 28,
                 max_model_len: int = 4096, gpu_memory_utilization: float = 0.85,
                 enable_prefix_caching: bool = True):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.engine_args = dict(
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt={"image": max(1, max_images_per_prompt)},
            mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
        )
        self.llm = LLM(model=model_name, dtype="bfloat16", **self.engine_args)
        self.processor = AutoProcessor.from_pretrained(model_name)  # 只用它的 chat template
        self.yes_id, self.no_id = yes_no_token_ids(self.llm.get_tokenizer())
        # 只允许 Yes/No：logprob 在 {Yes, No} 上重新归一化，但两者之差和全词表上完全相同
        self.sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=2,
                                       allowed_token_ids=[self.yes_id, self.no_id])

    def meta(self) -> dict:
        return {**super().meta(), **self.engine_args}

    def score_messages(self, msgs: list[Messages]) -> np.ndarray:
        from qwen_vl_utils import process_vision_info

        inputs = []
        for m in msgs:
            text = self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            images, _ = process_vision_info([m])  # 和 HF 后端同一套 resize
            item = {"prompt": text}
            if images:
                item["multi_modal_data"] = {"image": images}
            inputs.append(item)
        outs = self.llm.generate(inputs, self.sampling, use_tqdm=False)
        scores = np.empty(len(outs), dtype=np.float32)
        for i, o in enumerate(outs):
            lp = o.outputs[0].logprobs[0]
            if self.yes_id not in lp or self.no_id not in lp:
                raise RuntimeError(f"第 {i} 个 prompt 没拿到 Yes/No 的 logprob: {list(lp)}")
            scores[i] = lp[self.yes_id].logprob - lp[self.no_id].logprob
        return scores


def make_scorer(backend: str, **kw) -> _Base:
    return {"hf": QwenScorerHF, "vllm": QwenScorerVLLM}[backend](**kw)
