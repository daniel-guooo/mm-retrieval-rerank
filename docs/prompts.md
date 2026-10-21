# 提示词模板（zero-shot 重排用）

代码：`src/mmrr/models/qwen_prompt.py`（模板）、`src/mmrr/data/prompt_dataset.py`（prompt dataloader）。

## 参考来源

| 来源 | 借鉴了什么 |
|---|---|
| 快手 OpenOneRec [`data/onerec_data/sft/label_pred.py`](https://github.com/Kuaishou-OneRec/OpenOneRec)（commit `a969edc`） | pointwise 是/否判断任务的结构：system 角色说明 + 用户历史区块 + 针对候选的判断题 |
| [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) `data.py`（commit `5263062`） | Alpaca 式 `### Instruction / ### User Input / ### Response`；"in chronological order"；标题加引号 |

两者原本都用 SID（语义 ID）表示物品、训练时随机抽句式。我们改成：物品用 **标题（+图片）** 表示，推理时 **模板固定**。

## 模板

默认 `onerec`，`minionerec` 用来做 prompt 敏感性对照。示例（MicroLens，默认设置：历史只放标题，候选放标题 + 1 张图）：

**onerec@v1**
```
<system>
You are a personalized recommendation system. Based on a user's interaction history, you predict whether the user will be interested in a new video.

<user>
The user has watched the following videos in chronological order (oldest first):
1. "Street food tour in Chengdu"
2. "How to make dumplings at home"
3. "Cat reacts to cucumber"
4. "Spicy hotpot challenge"
5. "Night market snacks in Taipei"

Candidate video: "Making Sichuan mapo tofu"
[IMAGE: microlens/images/42.jpg]
Will the user watch the candidate video next? Answer Yes or No.
```

**minionerec@v1**
```
<user>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given the user's interaction history, will the user watch the candidate video next? Answer Yes or No.

### User Input:
The user has interacted with the following videos in chronological order: "Street food tour in Chengdu",	"How to make dumplings at home", ...
Candidate video: "Making Sichuan mapo tofu"
[IMAGE: microlens/images/42.jpg]
### Response:
```

## 规则

- 物品名词随数据集变：MicroLens 用 `video`，MovieLens / Netflix 用 `movie`。
- 每个物品"标题在前、图片在后"（vLLM 前缀缓存安全，见代码注释）。
- 缺图的物品只放标题。标题里的双引号换成单引号。
- `--history-images` 打开后历史 5 个物品也各带 1 张图（每个 prompt 6 张图，更慢）。
- 改任何模板文字都要升版本号；结果的 meta.json 里会记录 `onerec@v1` 这样的 id。
- 看真实数据上的 prompt：`python scripts/preview_prompts.py --split val --n 2`。

## prompt dataloader 用法

```python
from mmrr.data.prompt_dataset import RerankPromptDataset
ds = RerankPromptDataset(records, catalog, PromptConfig(template="onerec"), dataset="microlens")
len(ds)                                   # 用户数 × 50
ds[0]                                     # PromptExample(user, candidate, position, label, messages)
for recs, examples in ds.iter_user_batches(32):
    scores = scorer.score_messages([e.messages for e in examples])
```

`PromptExample` 可以直接放进 `torch.utils.data.DataLoader(ds, collate_fn=list)`。
