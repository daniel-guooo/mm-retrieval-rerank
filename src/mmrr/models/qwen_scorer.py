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


def qwen_prompt_inputs(processor, messages):
    """Use identical multimodal preprocessing at training and scoring time."""
    from qwen_vl_utils import process_vision_info

    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages]
    images, _ = process_vision_info(messages)
    return processor(text=text, images=images, padding=True, return_tensors="pt")


def load_qwen(model_name, device, precision, revision=None, sft_adapter=None):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, revision=revision, torch_dtype=dtypes[precision],
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    if sft_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, sft_adapter).merge_and_unload(safe_merge=True)
    return model.to(device), processor


def yes_no_margin(lm_head_weight, yes_id: int, no_id: int):
    """hidden @ margin == logit_Yes − logit_No，全程 FP32。"""
    return lm_head_weight[yes_id].float() - lm_head_weight[no_id].float()


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

    def __init__(self, model_name: str = DEFAULT_QWEN, device: str | None = None, batch_size: int = 8,
                 revision=None, precision=None, adapter=None, sft_adapter=None):
        import torch

        from mmrr.models.clip_encoder import pick_device

        self.model_name, self.batch_size = model_name, batch_size
        self.device = device or pick_device()
        self.precision = precision or {"cuda": "bf16", "mps": "fp16"}.get(self.device, "fp32")
        self.model, self.processor = load_qwen(model_name, self.device, self.precision, revision, sft_adapter)
        self.yes_id, self.no_id = yes_no_token_ids(self.processor.tokenizer)
        self.hf_commit = getattr(self.model.config, "_commit_hash", None)
        # BF16 logits 在 ~20 附近步长 0.125，大量候选会同分。分母抵消后 logP(Yes) − logP(No) = logit_Yes − logit_No，
        # 所以 lm_head 换成恒等（权重和 embedding 共享，embedding 不受影响），在 FP32 里只算 Yes−No 方向
        self.margin = yes_no_margin(self.model.lm_head.weight.detach(), self.yes_id, self.no_id)
        self.adapter, self.sft_adapter, self.revision = adapter, sft_adapter, revision
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        base = self.model.get_base_model() if adapter is not None else self.model
        base.lm_head = torch.nn.Identity()

    def meta(self) -> dict:
        return {**super().meta(), "hf_commit": self.hf_commit, "batch_size": self.batch_size,
                "revision": self.revision, "precision": self.precision,
                "adapter": self.adapter, "sft_adapter": self.sft_adapter,
                "score_precision": "fp32 hidden @ (W_yes - W_no)"}

    def score_messages(self, msgs: list[Messages]) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(msgs), self.batch_size):
                chunk = msgs[i:i + self.batch_size]
                inputs = qwen_prompt_inputs(self.processor, chunk).to(self.device)
                hidden = self.model(**inputs, use_cache=False).logits[:, -1, :]
                out.append((hidden.float() @ self.margin).cpu().numpy())
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
