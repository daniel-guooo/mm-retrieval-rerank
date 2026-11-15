# MicroLens 后半程运行手册

所有命令从小组仓库根目录运行。笔记本可以复算归档结果、准备数据和跑测试。完整模型运行目标是 AWS、GCP 或实验室电脑上的 **Linux + NVIDIA 单卡**。云机选择带 NVIDIA 驱动的镜像，并把数据和产物放在持久磁盘。实验室和云端使用相同的命令，只需调整磁盘路径、可见 GPU 和 batch size。

在远端准备包含本次改动的仓库副本，以及 `data-mllmrec.zip`。已推送代码时可检出对应提交。仍在本地工作区时，用文件传输工具上传包含新增文件的副本。已有的 Qwen 快照和 CLIP 缓存也可一并复制。运行后取回 `outputs/final/` 中的结果、清单与所需 adapter。

## 1. 环境和输入

Linux Bash 示例。可将 `/mnt/mmrr` 改成实际的持久路径，要求该路径可写。

```bash
export WORK=/mnt/mmrr
mkdir -p "$WORK"
unzip -q /path/to/data-mllmrec.zip -d "$WORK"
export MMRR_DATA_ROOT="$WORK/data"
export MMRR_OUTPUT_ROOT="$WORK/outputs"
export HF_HOME="$WORK/hf-cache"
export CUDA_VISIBLE_DEVICES=0
export QWEN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-final.txt
python -m pip install -e .
nvidia-smi
python scripts/pipeline.py preflight --gpu
python -m pytest -q
```

`MMRR_DATA_ROOT` 应直接包含 `microlens/`。Qwen 默认固定在 `66285546d2b821cf421d4f5eb2576359d3770cd3`。已有该版本完整快照时，把 `QWEN_MODEL` 改为快照目录。实验室交接中的本地快照目录和缓存可以复用。若使用其他权重 revision，所有 Qwen 命令应填写同一个 `--revision`。模型和缓存完整进入本地磁盘后可设置 `HF_HUB_OFFLINE=1`。

首选 BF16。预检显示设备不支持 BF16 时，可用 `--precision fp16` 在同一套试跑中验收，并在整组正式对照中保持精度一致。量产尺寸需根据实测显存及吞吐决定。见 [PyTorch 2.5.1 安装命令](https://pytorch.org/get-started/previous-versions/) 和 [Transformers 4.49 Qwen2.5-VL 接口](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_5_vl)。

### CPU 上复算已有结果

以下操作只需要 NumPy 和 Pillow。在 Windows PowerShell 中使用本机的 Python，给仓库执行一次 `python -m pip install -e .` 后，可逐条运行对应的 `pipeline.py` 命令。归档复算和融合无需解压数据或下载模型。

```bash
python -m pip install -e . numpy pillow pytest
V=manifests/microlens/val_interest_max_last_025_a0p5_top100.jsonl
ARCHIVE=reports/rerank/interest100_hits/val.jsonl
python scripts/pipeline.py evaluate --manifest "$V" --scores "$ARCHIVE" --catalog-size 19220 --output outputs/archive_check
python scripts/pipeline.py fuse --manifest "$V" --scores "$ARCHIVE" --output outputs/archive_fusion.jsonl
python -m pytest -q tests/test_final_pipeline.py
```

评估应有 5,000 名用户、499 个召回命中，并复现 HR@10 的 3.02% 和 2.44%。融合权重在 val 选择，结果保存在融合文件旁的 `.meta.json`。同一候选上的已有分数也可用 `evaluate --k 50` 复算 Top-50。

## 2. 单卡试跑

准备 CLIP 缓存。若复制队友缓存，放到 `$MMRR_OUTPUT_ROOT/emb/clip-vit-b32/microlens/`，应有 `item_ids.npy`、`item_text.npy`、`item_img.npy`、`item_has_img.npy` 和 `meta.json`。自行重建时，记录模型目录或 revision，以及新的缓存哈希。编码记录包含编码耗时和缓存字节数。

```bash
python scripts/encode_items.py --dataset microlens --metadata-only --img-bs 32
```

先用 16 名训练用户、4 名验证用户和两步参数更新检查整个链路。后续可扩大到 500–1,000 名训练用户估算吞吐。

```bash
SMOKE="$MMRR_OUTPUT_ROOT/smoke"
python scripts/pipeline.py candidates --split train --max-users 16 --output "$SMOKE/train.jsonl"
python scripts/pipeline.py prepare --manifest "$SMOKE/train.jsonl" --output "$SMOKE/examples"
python scripts/pipeline.py candidates --split val --max-users 4 --k 10 --output "$SMOKE/val.jsonl"
python scripts/pipeline.py train --stage sft --examples "$SMOKE/examples/random.jsonl" --model "$QWEN_MODEL" \
  --batch-size 1 --accumulation 1 --max-steps 2 --save-steps 1 --output "$SMOKE/sft"

PARENT="$SMOKE/sft/adapter"
python scripts/pipeline.py train --stage dpo --examples "$SMOKE/examples/random.jsonl" --model "$QWEN_MODEL" \
  --sft-adapter "$PARENT" --batch-size 1 --accumulation 1 --max-steps 2 --save-steps 1 --output "$SMOKE/dpo"

python scripts/pipeline.py hars-neighbors --model "$QWEN_MODEL" \
  --smoke-examples "$SMOKE/examples/random.jsonl" --output "$SMOKE/neighbors.npz"
python scripts/pipeline.py hars-attach --examples "$SMOKE/examples/random.jsonl" \
  --neighbors "$SMOKE/neighbors.npz" --output "$SMOKE/random_hars.jsonl"
python scripts/pipeline.py train --stage hanorec --examples "$SMOKE/random_hars.jsonl" --model "$QWEN_MODEL" \
  --sft-adapter "$PARENT" --batch-size 3 --accumulation 1 --max-steps 2 --save-steps 1 --output "$SMOKE/hanorec"

for stage in sft dpo hanorec; do
  parent=()
  if [ "$stage" != sft ]; then parent=(--sft-adapter "$PARENT"); fi
  python scripts/pipeline.py score --manifest "$SMOKE/val.jsonl" --model "$QWEN_MODEL" \
    --adapter "$SMOKE/$stage/adapter" "${parent[@]}" --shard-users 2 --output "$SMOKE/score_$stage"
  python scripts/pipeline.py evaluate --manifest "$SMOKE/val.jsonl" --scores "$SMOKE/score_$stage/scores.jsonl" \
    --output "$SMOKE/eval_$stage"
done
```

试跑通过需看到三个 adapter 加载成功、每个重排完整保存四条记录、训练日志和显存数据已写入。HaRS 真正的 batch 统计至少需要三个样本，梯度累积的步数单独配置。HaRS 试跑专用邻域只用于验收代码路径，正式训练使用全库邻域。正式跑前可把 `--shard-users` 调小以减轻中断时重跑的成本。

## 3. 正式训练与验证

主实验固定原始自然验证清单。训练候选与验证候选都用同样的全库兴趣公式。可先重建一份验证清单核对；新硬件重编码时，记录因浮点精度或 CLIP 版本导致的变化。正式实验统一使用同一份验证清单。

```bash
R="$MMRR_OUTPUT_ROOT/final"
V=manifests/microlens/val_interest_max_last_025_a0p5_top100.jsonl
ARCHIVE=reports/rerank/interest100_hits/val.jsonl
python scripts/pipeline.py candidates --split train --output "$R/train.jsonl"
python scripts/pipeline.py evaluate --manifest "$V" --catalog-size 19220 --popularity --output "$R/val/clip"
python scripts/pipeline.py evaluate --manifest "$V" --scores "$ARCHIVE" --output "$R/val/zero_shot"
python scripts/pipeline.py fuse --manifest "$V" --scores "$ARCHIVE" --output "$R/fusion_val.jsonl"
python scripts/pipeline.py evaluate --manifest "$V" --scores "$R/fusion_val.jsonl" --output "$R/val/fusion"
```

下面以一个种子演示，建议先跑完这个种子再用 `2026`、`2027` 扩展。默认四个负样本，召回组从原始名次 1–50 中采样。两组均排除训练序列内全部已观测物品。统一跳过不足四个合格负样本的用户并记录数量。未观测候选是训练用负例。

```bash
SEED=2025
E="$R/examples_$SEED"
python scripts/pipeline.py prepare --manifest "$R/train.jsonl" --seed "$SEED" --output "$E"
for neg in random retrieved; do
  NAME="sft_${neg}_$SEED"
  python scripts/pipeline.py train --stage sft --examples "$E/$neg.jsonl" --model "$QWEN_MODEL" \
    --seed "$SEED" --batch-size 1 --accumulation 16 --output "$R/$NAME"
  python scripts/pipeline.py score --manifest "$V" --model "$QWEN_MODEL" --hits-only \
    --sft-adapter "$R/$NAME/adapter" --output "$R/scoring/$NAME"
  python scripts/pipeline.py evaluate --manifest "$V" --scores "$R/scoring/$NAME/scores.jsonl" \
    --output "$R/val/$NAME"
done
```

正式 HaRS 邻域对各负采样种子可复用。Qwen 按原实现用标题 token 嵌入和视觉 encoder 特征平均池化，归一化后融合。分别取每个物品的得分最高的十个邻居，HaRS 计算正负物品的排序后近邻 softmax 分布之间的距离。当前实现是原作者方法在本项目 pointwise 协议下的适配。源实现见 [HaNoRec](https://github.com/wangyu0627/HaNoRec)。

```bash
python scripts/pipeline.py hars-neighbors --model "$QWEN_MODEL" --output "$R/neighbors.npz"
PARENT="$R/sft_random_$SEED/adapter"
# 为两种偏好方法选择相同的 batch 和累积步数
BATCH=3
ACCUM=8
for neg in random retrieved; do
  python scripts/pipeline.py hars-attach --examples "$E/$neg.jsonl" \
    --neighbors "$R/neighbors.npz" --output "$R/${neg}_hars_$SEED.jsonl"
  for stage in dpo hanorec; do
    EXAMPLES="$E/$neg.jsonl"
    if [ "$stage" = hanorec ]; then EXAMPLES="$R/${neg}_hars_$SEED.jsonl"; fi
    NAME="${stage}_${neg}_$SEED"
    python scripts/pipeline.py train --stage "$stage" --examples "$EXAMPLES" --model "$QWEN_MODEL" \
      --sft-adapter "$PARENT" --seed "$SEED" --batch-size "$BATCH" --accumulation "$ACCUM" --output "$R/$NAME"
    python scripts/pipeline.py score --manifest "$V" --model "$QWEN_MODEL" --hits-only \
      --sft-adapter "$PARENT" --adapter "$R/$NAME/adapter" --output "$R/scoring/$NAME"
    python scripts/pipeline.py evaluate --manifest "$V" --scores "$R/scoring/$NAME/scores.jsonl" \
      --output "$R/val/$NAME"
  done
done
python scripts/pipeline.py report --matrix configs/final_experiments.json --results "$R/val" --output "$R/validation.csv"
python scripts/pipeline.py compare --baseline "$R/val/sft_random_$SEED/ranks.jsonl" \
  --variant "$R/val/sft_retrieved_$SEED/ranks.jsonl" --output "$R/sft_paired_$SEED.json"
```

SFT 使用回答位置的全词表交叉熵。DPO 和 HaNoRec 使用相同的单 token preference 目标，参考模型是合并后的冻结 SFT。偏好 adapter 在该参考基础上新建。NoDO 在前向与梯度检查点重算中使用同一份噪声。默认 LoRA 对语言层 attention/MLP 投影生效，视觉 encoder 冻结。默认每阶段一轮训练，先根据吞吐和 val 确定正式预算，组内保持一致。当前适配独立于参考仓库缺失的定制 LLaMA-Factory。

## 4. 恢复运行与最终结果

- **训练中断**：同一条 `train` 命令加 `--resume`。恢复会寻找最新的完整 checkpoint，并核对原有配置和数据指纹。配置需要改变时，用新的输出目录。训练的最终可分发产物位于 `adapter/`。
- **重排中断**：用原样的 `score` 命令和输出目录重新启动，已完成分片会被核验并复用。完成后输出 `scores.jsonl`。使用 `--hits-only` 可精确还原固定候选的 HR/NDCG，效率用独立的全查询抽样测量。
- **最终测试**：先记录 val 确定的召回、K、模型、adapter、prompt 和融合权重，再用 `candidates --split test --output "$R/test.jsonl"` 生成自然候选。主对照共用该文件。评估时把输出放到 `"$R/test/<实验名>"`。对选中的融合策略，测试命令显式传入验证得到的 `--weight`。
- **补充对照**：在 val 用 `score --no-candidate-images` 遮蔽候选图，用 `--history-images` 加历史图，或用 `--template minionerec` 对照提示词。训练负样本难度比较可在 `prepare` 中使用 `--rank-low 1 --rank-high 10` 和 `--rank-low 41 --rank-high 50`。对照时核对保留的用户集合与样本数。
- **原版 HaNoRec 复现**：原版 AUC 和小候选池排序需要其原任务格式、训练集成和 checkpoint。其进度记录在报告中，当前主流水线评估的是小组的全库召回任务。

```bash
python scripts/pipeline.py benchmark --model "$QWEN_MODEL" --k 50 --output "$R/timing_k50.json"
python scripts/pipeline.py benchmark --model "$QWEN_MODEL" --k 100 --output "$R/timing_k100.json"
python scripts/pipeline.py recommend --history 1 2 3 --clip-only
```

测速输出 CPU 召回和 HF 重排延迟、吞吐和 GPU 峰值分配。测 adapter 时加上与 `score` 相同的 adapter 参数。测算用同一硬件串行执行。正式结果用 [结果模板](../reports/final_results_template.md) 汇总。建议保存模型标识、运行配置与 sidecar、分片或完整评分、`ranks.jsonl`、CSV、计时文件和可分发的 adapters。目录中的数据和各阶段中间产物留在持久磁盘。

可选 FAISS 演示。在兼容的 Linux 环境中安装 `faiss-cpu` 后，给 `recommend` 或 `benchmark` 添加 `--faiss-depth 400`。该入口对每条历史分别用 `IndexFlatIP` 查询，再对候选并集用原兴趣公式重打分。计时文件会额外记录与精确候选的重合率和抽样命中数。正式候选默认使用精确搜索。
