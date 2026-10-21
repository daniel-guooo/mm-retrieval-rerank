# mm-retrieval-rerank

两阶段多模态推荐：Stage 1 冻结 CLIP 全库召回 Top-50 → Stage 2 Qwen2.5-VL-3B pointwise 重排。

## 环境

```bash
conda create -p ./mm_retrieval_rerank python=3.10 -y
conda activate ./mm_retrieval_rerank
pip install -r requirements.txt          # GPU 机器（Linux/WSL2）用 requirements-vllm.txt
pip install -e .
```

数据和输出放哪由环境变量决定，见 [docs/interfaces.md](docs/interfaces.md)。

## 目录

```
src/mmrr/data/     数据接口（schema.py 约定格式，loaders.py 待实现）+ prompt_dataset.py（拼提示词）
src/mmrr/models/   CLIP 编码器、提示词模板 qwen_prompt.py、Qwen 打分（HF / vLLM）
src/mmrr/eval/     名次与记录格式
scripts/           check_loader / encode_items / preview_prompts / rerank_qwen
manifests/         召回候选（进 git）
docs/interfaces.md 所有人对齐的接口约定
docs/prompts.md    提示词模板说明和示例
```
