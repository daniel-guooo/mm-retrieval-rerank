# 接口约定（v0）

改这里的任何约定，先在群里说，再改代码和本文件。

## 1. 路径

- 代码里**不写绝对路径**。所有路径从 `src/mmrr/paths.py` 取。
- 每台机器用环境变量指定本机位置（不设就用 repo 下的默认目录）：

  | 变量 | 默认 | 放什么 |
  |---|---|---|
  | `MMRR_DATA_ROOT` | `<repo>/data` | 解压后的数据包 |
  | `MMRR_OUTPUT_ROOT` | `<repo>/outputs` | 向量缓存、重排结果（不进 git） |

- `manifests/` 在 repo 里、进 git，所有人共用。
- 文件里只存 **item id**，不存图片路径；图片路径由每台机器的 `load_catalog` 解析。所以同一个 jsonl 换机器直接能用。

## 2. dataloader（数据负责人实现）

实现 `src/mmrr/data/loaders.py` 里的两个函数，数据结构见 `src/mmrr/data/schema.py`：

```python
load_catalog(dataset) -> dict[int, Item]        # Item(item_id: int, title: str, image_path: str)
load_sequences(dataset, split) -> list[Sequence] # Sequence(user: str, history: list[int], target: int)
```

| 字段 | 约定 |
|---|---|
| `dataset` | `"microlens"` / `"movielens"` / `"netflix"`（小写） |
| `split` | `"train"` / `"val"` / `"test"`，用官方划分 |
| `item_id` | int，数据集原始 id |
| `title` | 原始标题，不截断 |
| `image_path` | **相对 `MMRR_DATA_ROOT`** 的 posix 路径，如 `microlens/images/123.jpg`；**没有图就填 `None`**（不要像 HaNoRec 那样用别的商品的图顶替） |
| `user` | str |
| `history` | 最近的交互，**从旧到新**，长度 1..5 |
| `target` | 紧接 history 的下一次交互 |
| catalog 范围 | 全库所有商品（召回是全库检索） |
| 顺序 | 多次调用结果一致 |

交付标准：`python scripts/check_loader.py --dataset microlens` 输出 `OK`。

### 2.1 原始文件格式（从 HaNoRec `utils/data_loader.py` 反推，下载后需核实）

| 文件 | 格式 |
|---|---|
| `data/<dataset>/{train,val,test}.tsv` | 无表头、tab 分隔；第 0 列 user_id（int），第 1 列空格分隔的 item id 序列 |
| `data/<dataset>/<Dataset>_titles.csv` | 有表头，列 `item`、`title` |
| `data/<dataset>/images/<item_id>.jpg` | 封面图；**有缺图**（HaNoRec 缺图时拿 `627.jpg` 顶替） |

HaNoRec 的做法：序列长度 < 6 的直接丢掉；取最后 6 个，前 5 个是 history，最后 1 个是 target；item id 范围 1..max_item_id。
我们是否也丢掉短序列，需要和数据负责人定（见下）。

## 3. 召回候选（Stage 1 产出，进 git）

`manifests/<dataset>/<split>_top50.jsonl`，一行一个用户：

```json
{"user": "u1", "history": [1,2,3,4,5], "target": 456,
 "candidates": [/* 50 个 id，按分数降序 */], "scores": [/* 50 个 */], "target_rank": 3}
```

- `target_rank` 从 1 开始；没召回到为 `null`。不手动把 target 塞进候选。
- 候选里已排除 history 里的商品。
- 旁边的 `<split>_top50.meta.json` 记录 α、γ、聚合方式、CLIP 版本、git commit。
- 生成后冻结，改了要通知所有人。

## 4. CLIP 向量缓存

`$MMRR_OUTPUT_ROOT/emb/clip-vit-b32/<dataset>/`：`item_ids.npy` (N,) int64、`item_text.npy` / `item_img.npy` (N,512) float32 已 L2 归一化，三者行对齐；`meta.json` 记录 HF commit 和被截断的标题数。

## 5. Qwen 重排结果

`$MMRR_OUTPUT_ROOT/rerank/<tag>/<dataset>/<split>.jsonl`：

```json
{"user": "u1", "target": 456,
 "candidates": [/* 和候选文件完全一样的顺序 */],
 "scores": [/* log P(Yes) − log P(No)，和 candidates 逐位对应 */],
 "reranked": [/* 按 scores 降序，分数相同保持召回顺序 */],
 "target_rank_retrieval": 3, "target_rank_rerank": 1}
```

`<split>.meta.json` 记录模型 commit、prompt 版本与图片设置、Yes/No token id、后端参数、总耗时、每用户耗时、每用户 Qwen 调用次数。

## 6. 打分和 prompt

- pointwise：每个用户 50 个候选 → 50 个 prompt（history + 1 个候选）→ 50 个分数 → 排序。
- 分数取答案位置下一个 token：`s = log P(Yes) − log P(No)`，不采样，结果确定。
- 模板和 prompt dataloader 见 [prompts.md](prompts.md)。

## 7. 指标约定

- 端到端 HR/NDCG 在**全部** query 上算，target 没召回到记 0。
- 另报只在 "target 被召回" 的 query 上的条件指标。
- 超参只在 val 上调；test 只跑一次。
