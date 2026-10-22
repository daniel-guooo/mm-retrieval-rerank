# Qwen2.5-VL-3B zero-shot 重排：interest100（验证集）

全部为 MicroLens **验证集**结果，不是测试集。召回方案 interest100（75%×历史最高相似度＋25%×最近一次，α=0.5，K=100）也是在验证集上按 Recall 选出的。

## 设置

- 模型 Qwen2.5-VL-3B-Instruct（revision `66285546d2b821cf421d4f5eb2576359d3770cd3`），HF 后端，BF16，候选 batch=1
- 提示词 `onerec@v1`：历史标题（最多5项，不放图）＋候选标题和图片，像素 50,176～200,704
- 分数 `logP(Yes) − logP(No)`，在 FP32 中由最后一层隐藏状态与 `W_Yes − W_No` 点积得到（BF16 logits 会造成大量并列分数），降序排序
- 候选由 `scripts/make_oracle_candidates.py` 从 `val_interest_max_last_025_a0p5_top100.jsonl` 拆分：
  - **A（hits）**：召回含 target 的全部 499 名用户，候选不改动
  - **B（inject）**：召回不含 target 的 4501 名用户，删去第100名，target 随机插入 1..100（seed 2025）；重排时再随机抽 500 名（seed 2025）
- 输出元数据记录的 `git_commit` 为 `b098640`；运行时 FP32 打分改动尚未提交，已随本报告一同提交

## A：完整 pipeline

未命中用户重排后指标必为 0，所以全部 5000 名用户上的指标 = Recall@100 (9.98%) × 命中用户上的指标，与跑满 5000 人数学上相同。

| 指标 | 命中用户：CLIP 顺序 | 命中用户：Qwen 重排 | 全部用户：CLIP | 全部用户：Qwen |
|---|---:|---:|---:|---:|
| HR@3 | 0.1283 | 0.0822 | 0.0128 | 0.0082 |
| NDCG@3 | 0.1010 | 0.0633 | 0.0101 | 0.0063 |
| HR@10 | 0.3026 | 0.2445 | 0.0302 | 0.0244 |
| NDCG@10 | 0.1610 | 0.1204 | 0.0161 | 0.0120 |

- target 名次：237 人上升、253 人下降、9 人不变；中位名次 26 → 30；符号检验 z = −0.72（整体名次变化不显著）
- 重排后 HR@10 的 bootstrap 95% CI：0.206～0.283
- Qwen 平均把 target 排在约 65.5% 的候选之前（随机 50%），CLIP 约 66.3%：有信号，但没有超出 CLIP

## B：模型本身的重排能力（GT 随机注入，500 人）

| 指标 | 随机插入位置 | Qwen 重排 | 随机期望 |
|---|---:|---:|---:|
| HR@3 | 0.014 | 0.034 | 0.03 |
| NDCG@3 | 0.0115 | 0.0233 | — |
| HR@10 | 0.082 | 0.094（95% CI 0.068～0.120） | 0.10 |
| NDCG@10 | 0.0350 | 0.0435 | — |
| target 中位名次 | 53 | 65 | 50.5 |

- 198 人上升、299 人下降、3 人不变；符号检验 z = −4.53：Qwen 显著把注入的 target 排得比随机更靠后
- 解读：这些 target 正是 CLIP 判断与历史不相似的物品，而另外 99 个候选是 CLIP 选出的最相似物品（难负样本）。zero-shot Qwen 的判断主要也依据与历史的相似度，因此会系统性地把它们往后排。B 测的是“相似度信号失效时模型能否找回 target”，不应直接解读为“Qwen 不如随机”

## 结论

zero-shot Qwen2.5-VL-3B 本质上在做与 CLIP 相近的相似度判断，没有带来提升，在前 3 和前 10 名上弱于 CLIP 原顺序。下一步可考虑 CLIP 与 Qwen 分数融合，或用 SFT 让模型学习数据集中的用户行为。

## 耗时

A、B 两组在同一张 RTX 4090 上**并行**运行，记录的 19.9 秒/用户（每用户 100 次打分）因共享 GPU 偏慢，不代表单任务速度。单任务 Top-50 实测为 4.44 秒/用户。

## 文件

- `interest100_hits/`、`interest100_inject500/`：`val.jsonl`（每用户候选、Qwen 分数、重排结果）、`val.meta.json`、`val.metrics.json`
- 候选：`manifests/microlens/val_interest_max_last_025_a0p5_top100_{hits,inject}.jsonl`

复现：

```bash
M=manifests/microlens/val_interest_max_last_025_a0p5_top100
python scripts/make_oracle_candidates.py $M.jsonl --mode hits
python scripts/make_oracle_candidates.py $M.jsonl --mode inject
python scripts/rerank_qwen.py --backend hf --k 100 --candidates ${M}_hits.jsonl --model "$QWEN_MODEL" --hf-batch-size 1 --users-per-call 1 --tag interest100_hits
python scripts/rerank_qwen.py --backend hf --k 100 --candidates ${M}_inject.jsonl --model "$QWEN_MODEL" --hf-batch-size 1 --users-per-call 1 --max-users 500 --tag interest100_inject500
python scripts/evaluate_rerank.py "$MMRR_OUTPUT_ROOT/rerank/<tag>/microlens/val.jsonl"
```
