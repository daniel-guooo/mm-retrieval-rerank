# Qwen 重排交接：实验室电脑执行指南

交接日期：2026-10-21。目标：使用已冻结的六组 MicroLens 验证集候选，运行 Qwen2.5-VL-3B-Instruct **zero-shot 推理**，比较 HR@3/10、NDCG@3/10 和耗时。无需训练、重新编码或重新检索。

## 1. 已完成与尚未完成

- 已完成数据清理、19,220 个物品的 CLIP 编码、5,000 条验证样本检索、9组参数搜索、两组多路召回。
- 六份候选均已核验；参数与 SHA256 集中在 `configs/qwen_handoff.json`，详细比较在 `reports/retrieval/`。
- **尚未在实验室机器实际运行 Qwen**。启动接口使用模拟评分器测试；这不代表 GPU 推理已验证。首先完成下述真实20用户试跑。
- 现在所有候选都是 **val**。不要写成 test 结果。最终测试需先固定方案，再另外生成 test 候选。

## 2. GitHub中有什么，另外要传什么

GitHub包含代码、环境依赖、六份候选及其元数据、全部检索参数比较和本指南。六份JSONL合计约47.8 MB；不用 Git LFS。模型、原始图片、embedding、Python虚拟环境不上传。

另行传给同学这两个目录，或在实验室重新下载相同版本：

1. `D:\HaNoRec\data-mllmrec\data\microlens`：最简单是整个文件夹直接复制。重排加载器至少需要 `Microlens_titles.csv`、`train.tsv`、`val.tsv`、`test.tsv`、`images/`。已有的 `*-mllm_sft_*.json` 和 pairs CSV 不用于这次推理。保留原物品ID、文件名和图片，不重新划分数据。
2. `D:\HaNoRec\models\Qwen2.5-VL-3B-Instruct`：整个模型目录，含权重分片、索引、config、tokenizer、processor等。仅复制权重不够。模型版本固定为 `66285546d2b821cf421d4f5eb2576359d3770cd3`。

**只跑Qwen不需要CLIP模型，也不需要embedding。** 若还要在实验室调召回参数，再复制 `outputs/emb/clip-vit-b32/microlens/` 中四份 `.npy` 和 `meta.json`。不要复制 Windows `.venv` 到另一台机器。

## 3. 六组实验与固定设置

| 启动实验名 | 候选文件（均在 manifests/microlens） | K | α | γ | Recall@K |
|---|---|---:|---:|---:|---:|
| baseline50 | val_top50.jsonl | 50 | 0.5 | 1 | 6.16% |
| baseline100 | val_top100.jsonl | 100 | 0.5 | 1 | 8.52% |
| best50 | val_fusion_a0p5_g0p5_top50.jsonl | 50 | 0.5 | 0.5 | 6.48% |
| best100 | val_fusion_a0p5_g0p5_top100.jsonl | 100 | 0.5 | 0.5 | 8.84% |
| multi50 | val_multi_f30_t10_i10_a0p5_g0p5_top50.jsonl | 50 | 0.5 | 0.5 | 6.40% |
| multi100 | val_multi_f60_t20_i20_a0p5_g0p5_top100.jsonl | 100 | 0.5 | 0.5 | 8.56% |

α是图文融合中的文本权重。γ是历史衰减系数，历史最多5项，从旧到新的权重是 γ^(m−1), …, γ, 1。

多路按图文→文本→图片分配30/10/10或60/20/20个**不同物品**，重复时从该路继续补足；选定集合后用RRF排序（常数60，每路深度100）。两种多路规模是独立方案，不要求50项是100项的前缀。

重排统一使用：`onerec@v1`，历史标题＋候选标题和图片，不开启历史图片；图片像素范围50,176～200,704；分数为第一个答案位置 `logP(Yes) − logP(No)`，降序排序。target只用于评估，不进入提示词。默认HF后端，BF16 CUDA，初始候选batch=1；随机子集seed=2025。所有组保持相同模型、模板、图像设置和评估样本。

## 4. 实验室环境（Linux + NVIDIA 示例）

仓库分支：`codex/qwen-lab-handoff`。

```bash
git clone --branch codex/qwen-lab-handoff https://github.com/daniel-guooo/mm-retrieval-rerank.git
cd mm-retrieval-rerank
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
nvidia-smi
python -c "import torch; print('torch:',torch.__version__,'CUDA:',torch.version.cuda,'available:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('BF16:',torch.cuda.is_bf16_supported())"
```

该组合沿用本项目现有PyTorch/Transformers版本；GPU驱动需支持所装CUDA运行时。若CUDA不可用、BF16不支持、安装失败，先处理环境，不要直接开全量。**本指南先使用HF后端，不安装 `requirements-vllm.txt`**；vLLM为另外的加速路线，需要单独验证依赖和评分一致性。

Windows机器可使用Python 3.10～3.12，创建 `.venv` 后用 `.\.venv\Scripts\python.exe` 替代命令里的 `python`；PowerShell激活是 `.\.venv\Scripts\Activate.ps1`。其余Python脚本命令相同。

## 5. 配置数据、模型和输出路径

下面是例子，请改成实验室实际路径。`MMRR_DATA_ROOT` 必须指向 **microlens 的上一层**。

```bash
export MMRR_DATA_ROOT=/mnt/data/hanorec/data
export MMRR_OUTPUT_ROOT=/mnt/data/hanorec/qwen_outputs
export QWEN_MODEL=/mnt/models/Qwen2.5-VL-3B-Instruct
export HF_HUB_OFFLINE=1
```

应存在 `/mnt/data/hanorec/data/microlens/images/`。离线模式只在模型完整复制后开启。PowerShell对应写法：

```powershell
$env:MMRR_DATA_ROOT = 'D:\HaNoRec\data-mllmrec\data'
$env:MMRR_OUTPUT_ROOT = 'D:\HaNoRec\qwen_outputs'
$env:QWEN_MODEL = 'D:\HaNoRec\models\Qwen2.5-VL-3B-Instruct'
$env:HF_HUB_OFFLINE = '1'
```

如果需要重新下载模型，先在联网环境执行，再开启离线模式：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', revision='66285546d2b821cf421d4f5eb2576359d3770cd3', local_dir='/mnt/models/Qwen2.5-VL-3B-Instruct')"
```

## 6. 先试跑20个用户，再测速100个用户

Linux命令：

```bash
python scripts/run_handoff.py --experiment best50 --model "$QWEN_MODEL" --max-users 20 --tag best50_smoke20
python scripts/evaluate_rerank.py "$MMRR_OUTPUT_ROOT/rerank/best50_smoke20/microlens/val.jsonl"
python scripts/run_handoff.py --experiment best50 --model "$QWEN_MODEL" --max-users 100 --tag best50_timing100
```

PowerShell对应第一条命令：

```powershell
python scripts/run_handoff.py --experiment best50 --model "$env:QWEN_MODEL" --max-users 20 --tag best50_smoke20
```

首次启动会完整检查图片并加载模型，可能数分钟没有推理进度。重排进度条出现后才进入模型评分。每个用户50次或100次候选评分；20用户Top-50是1,000次。输出meta中的 `seconds_total` 不含数据检查、模型加载，`seconds_per_user` 可估算5000用户评分时间。Top-100另外测一次，别直接承诺精确翻倍耗时。

先保持 `--hf-batch-size 1`；显存充足时，可用新的tag测速batch=2或4，记录显存与速度。`users-per-call` 与候选batch不同：本启动器前者固定1，后者控制一次前向的候选数量。不要开启 `--history-images`，否则改变实验设定并增加显存。

## 7. 全量六组实验

建议先跑best50，确认格式和评估正常后跑best100，再跑其余对照。Linux下每条单独运行：

```bash
python scripts/run_handoff.py --experiment best50 --model "$QWEN_MODEL" --tag best50_full
python scripts/run_handoff.py --experiment best100 --model "$QWEN_MODEL" --tag best100_full
python scripts/run_handoff.py --experiment baseline50 --model "$QWEN_MODEL" --tag baseline50_full
python scripts/run_handoff.py --experiment baseline100 --model "$QWEN_MODEL" --tag baseline100_full
python scripts/run_handoff.py --experiment multi50 --model "$QWEN_MODEL" --tag multi50_full
python scripts/run_handoff.py --experiment multi100 --model "$QWEN_MODEL" --tag multi100_full
```

去掉 `--max-users` 才是完整5000用户。三组Top-50各25万次评分，三组Top-100各50万次评分，总计225万次；当前实现不跨实验复用Qwen分数。

**当前脚本整次结束才写结果，不支持断点续跑。** 在稳定的终端会话运行；Linux可使用已有的tmux/screen会话。不要在共享GPU上同时启动多组模型。若OOM，停止该进程并降低候选batch；batch=1仍OOM则需要更大可用显存或另行验证量化方案。输出已有时脚本拒绝覆盖，要用新tag。

## 8. 评估与返还结果

每组完成后运行（将best50_full替换成对应tag）：

```bash
python scripts/evaluate_rerank.py "$MMRR_OUTPUT_ROOT/rerank/best50_full/microlens/val.jsonl"
python -m pip freeze > environment-lab.txt
nvidia-smi > gpu-lab.txt
git rev-parse HEAD > code-commit.txt
```

每组返还：`val.jsonl`、`val.meta.json`、`val.metrics.json`；另外返还GPU信息、依赖版本、代码提交号和实际使用的命令/日志。meta记录候选文件名及SHA256，便于防止实验串组。

评估分母必须是全部用户，正确目标未被召回的用户记0，不能只算命中的用户。全量 `n_users` 应为5000，`candidate_recall` 应与上表对应；重排只能改变候选顺序，不能提高候选集合召回率。HR@3/10与NDCG@3/10才是要比较的最终指标。原多路顺序是RRF，单路顺序是相似度，两者的重排前排名指标需要按各自顺序解释。

不要仅凭检索Recall就宣称Qwen改善，也不要将20/100用户试跑混入5000用户结果。

## 9. 参数复现入口

- `configs/qwen_handoff.json`：六组候选与固定推理设置。
- `scripts/run_handoff.py`：按实验名选择文件并核验哈希；`--dry-run`只显示最终命令。
- `scripts/rerank_qwen.py`：底层可用 `--candidates` 指定文件，`--hf-batch-size`控制候选batch。
- `reports/retrieval/retrieval-grid.csv`：9组α、γ的全部指标。
- `scripts/tune_retrieval.py`：复现参数搜索，需额外复制embedding。
- `scripts/evaluate_multiroute.py`：复现两组多路召回，需embedding。

只交付这六组已冻结候选即可开始Qwen；无需先重做上述召回实验。
