"""Single-GPU LoRA training for the first Yes/No answer token."""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, set_seed

from mmrr.artifacts import adapter_fingerprint, load_json, new_directory, require_new, save_json, sha256, verified_sidecar
from mmrr.data.loaders import load_catalog
from mmrr.io import read_jsonl
from mmrr.models.preference import AnswerHead, nodo_noise, preference_loss
from mmrr.models.qwen_prompt import PromptConfig, build_messages, yes_no_token_ids
from mmrr.models.qwen_scorer import load_qwen, qwen_prompt_inputs
from mmrr.paths import DATA_ROOT

ATTENTION_AND_MLP = r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def complete_checkpoint(output):
    required = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "rng_state.pth", "trainer_state.json")
    steps = [(int(path.name[11:]), path) for path in output.glob("checkpoint-*")
             if path.name[11:].isdigit() and all((path / name).is_file() for name in required)]
    return str(max(steps)[1]) if steps else None


class PointwiseCollator:
    def __init__(self, dataset, catalog, processor, prompt, max_tokens):
        self.dataset = dataset
        self.catalog = catalog
        self.processor = processor
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.answer_ids = dict(zip(("Yes", "No"), yes_no_token_ids(processor.tokenizer), strict=True))

    def messages(self, row):
        return build_messages([self.catalog[i] for i in row["history"]], self.catalog[row["candidate"]],
                              self.prompt, self.dataset)

    def check_answer_tokens(self, example):
        messages = self.messages(example)
        render = self.processor.apply_chat_template
        encode = self.processor.tokenizer.encode
        prefix = encode(render(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
        for answer, token in self.answer_ids.items():
            full = messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
            ids = encode(render(full, tokenize=False, add_generation_prompt=False), add_special_tokens=False)
            if ids[:len(prefix) + 1] != [*prefix, token]:
                raise ValueError(f"The contextual {answer} token does not match the inference tokenizer")

    def __call__(self, examples):
        inputs = qwen_prompt_inputs(self.processor, [self.messages(row) for row in examples])
        if inputs["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Prompt exceeds max tokens; increase the limit or explicitly change the prompt")
        inputs["labels"] = torch.tensor([self.answer_ids[row["chosen"]] for row in examples])
        if "hardness" in examples[0]:
            inputs["hardness"] = torch.tensor([row["hardness"] for row in examples], dtype=torch.float32)
        return inputs


class PointwiseTrainer(Trainer):
    def __init__(self, *args, stage, yes_id, beta, sigma, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage, self.yes_id, self.beta, self.sigma = stage, yes_id, beta, sigma
        self.model_accepts_loss_kwargs = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Trainer performs backward inside training_step, including checkpoint recomputation.
        with nodo_noise(self.accelerator.unwrap_model(model), self.sigma if self.stage == "hanorec" else 0):
            return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        hardness = inputs.pop("hardness", None)
        if self.stage == "hanorec" and hardness is None:
            raise ValueError("HaNoRec requires hardness for every example")
        if self.stage != "sft":
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                    reference = model(**inputs, use_cache=False).logits[:, -1, 0]
            finally:
                model.train(was_training)
        outputs = model(**inputs, use_cache=False)
        answers = outputs.logits[:, -1, :]
        if self.stage == "sft":
            loss = F.cross_entropy(answers.float(), labels)
        else:
            loss = preference_loss(answers[:, 0], reference, labels.eq(self.yes_id),
                                   self.beta, hardness if self.stage == "hanorec" else None)
        return (loss, outputs) if return_outputs else loss


def train_adapter(args):
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The training pipeline expects one visible NVIDIA GPU (set CUDA_VISIBLE_DEVICES)")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU cannot run BF16. Select an available GPU or use --precision fp16")
    if (args.stage == "sft") != (args.sft_adapter is None):
        raise ValueError("DPO and HaNoRec require --sft-adapter; SFT starts from the base model")
    if args.stage == "hanorec" and args.batch_size < 3:
        raise ValueError("HaRS requires --batch-size >= 3 per microbatch; gradient accumulation is separate")
    input_meta = verified_sidecar(args.examples)
    if input_meta["split"] != "train":
        raise ValueError("Training requires a prepared train artifact")
    dataset = input_meta["dataset"]
    if sha256(DATA_ROOT / dataset / "train.tsv") != input_meta["train_sha256"]:
        raise ValueError("train.tsv differs from the source used to build these examples")
    if args.stage == "hanorec" and not input_meta.get("neighbors_sha256"):
        raise ValueError("Generate and attach HaRS hardness before HaNoRec training")
    if args.stage == "hanorec" and input_meta["hars_scope"] == "smoke" and not 1 <= args.max_steps <= 20:
        raise ValueError("Smoke HaRS neighbors require a test run of at most 20 update steps")
    rows = list(read_jsonl(args.examples))
    if len(rows) != input_meta["n_examples"] or not rows:
        raise ValueError("Training example count does not match the artifact")
    if args.stage == "hanorec" and len(rows) < args.batch_size:
        raise ValueError("Need at least one full HaRS microbatch")
    if any({row["chosen"], row["rejected"]} != {"Yes", "No"}
           or (row["chosen"] == "Yes") != (row["candidate"] == row["positive_item"]) for row in rows):
        raise ValueError("Invalid pointwise training label")

    parent = adapter_fingerprint(args.sft_adapter)
    if args.sft_adapter is not None and (args.sft_adapter / "mmrr.json").is_file():
        parent_meta = load_json(args.sft_adapter / "mmrr.json")
        if (parent_meta["stage"] != "sft" or parent_meta["revision"] != args.revision
                or parent_meta["dataset"] != dataset):
            raise ValueError("Use a matching revision and the SFT adapter for this dataset")
    output = args.output
    adapter = output / "adapter"
    run_file = output / "run.json"
    config = {
        key: getattr(args, key) for key in ("stage", "model", "revision", "precision", "seed", "batch_size",
                                           "accumulation", "epochs", "max_steps", "save_steps", "lr", "rank",
                                           "lora_alpha", "lora_dropout", "beta", "sigma", "max_tokens")
    }
    prompt = PromptConfig()
    config.update({
        "example_sha256": input_meta["sha256"], "dataset": dataset, "strategy": input_meta["strategy"],
        "sampling_seed": input_meta["seed"], "sft_adapter_sha256": parent,
        "prompt": prompt.to_dict(), "n_examples": len(rows),
        "titles_sha256": sha256(DATA_ROOT / dataset / f"{dataset.capitalize()}_titles.csv"),
    })
    checkpoint = None
    if args.resume:
        if load_json(run_file)["config"] != config:
            raise ValueError("Resume requires the original training configuration and data")
        checkpoint = complete_checkpoint(output)
        if checkpoint is None:
            raise FileNotFoundError(f"No training checkpoint in {output}")
    else:
        require_new(run_file, adapter)
    require_new(adapter)

    started = time.perf_counter()
    set_seed(args.seed)
    model, processor = load_qwen(args.model, "cuda", args.precision, args.revision, args.sft_adapter)
    collator = PointwiseCollator(dataset, load_catalog(dataset, include_interaction_ids=False),
                                 processor, prompt, args.max_tokens)
    collator.check_answer_tokens(rows[0])
    yes, no = collator.answer_ids["Yes"], collator.answer_ids["No"]
    model.lm_head = AnswerHead(model.lm_head, yes, no, args.stage != "sft")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=ATTENTION_AND_MLP, task_type="CAUSAL_LM", bias="none",
    ))
    model.print_trainable_parameters()

    setup_seconds = time.perf_counter() - started
    arguments = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation, num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.lr, bf16=args.precision == "bf16",
        fp16=args.precision == "fp16", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        dataloader_drop_last=args.stage != "sft", dataloader_num_workers=0, optim="adamw_torch",
        lr_scheduler_type="cosine", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=10, report_to=[], seed=args.seed, data_seed=args.seed,
        save_safetensors=True,
    )
    trainer = PointwiseTrainer(
        model=model, args=arguments, train_dataset=rows, data_collator=collator,
        stage=args.stage, yes_id=yes, beta=args.beta, sigma=args.sigma,
    )
    if not args.resume:
        save_json(run_file, {"config": config, "examples": str(args.examples),
                             "sft_adapter": str(args.sft_adapter) if args.sft_adapter else None}, meta=True)
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    peak = torch.cuda.max_memory_allocated()
    with new_directory(adapter) as stage:
        trainer.save_model(str(stage))
        save_json(stage / "mmrr.json", config)
    save_json(output / "metrics.json", {
        **result.metrics, "seconds_setup": setup_seconds, "peak_allocated_bytes": peak,
        "gpu": torch.cuda.get_device_name(), "adapter_sha256": adapter_fingerprint(adapter),
    }, meta=True)
    print(f"Saved {args.stage} adapter: {adapter}", flush=True)
