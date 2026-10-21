# Validation retrieval tuning

Run from the repository after configuring `MMRR_DATA_ROOT` and `MMRR_OUTPUT_ROOT`:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tune_retrieval.py
```

This evaluates MicroLens validation queries using the existing frozen CLIP embeddings. No model training, image re-encoding, test-set tuning, or Qwen inference is performed.

Grid: alpha (text weight) = 0.25, 0.5, 0.75; gamma (older-history discount) = 0.5, 0.8, 1.0. Gamma 1 gives equal history weights. Primary selection is highest Recall@50, breaking ties by Recall@100, larger gamma, then alpha closest to 0.5. A separate winner is reported for Recall@100. These are the best configurations in this grid, not a claim of a global optimum.

Results: `outputs/retrieval/microlens/val/grid/metrics.csv`, `selection.json`, and `validation.json`. The full execution log may be redirected to `outputs/retrieval-grid.log`.

Original baseline manifests remain `manifests/microlens/val_top50.jsonl` and `val_top100.jsonl` (alpha 0.5, gamma 1). Distinct winning configurations are saved with a parameter tag, for example `val_fusion_a0p5_g0p8_top50.jsonl`. Both cutoffs are exported for each distinct winner. Metadata records parameters, source hashes, engine hash, and manifest hash. Existing manifests are never overwritten; a repeat grid run validates and reuses matching files.

Checks cover query/history/target alignment, candidate count and uniqueness, exclusion of input history, finite descending scores, target ranks, recall recomputation, source hashes, and identical Top-50 prefixes in corresponding Top-100 files. The original baseline Top-50 hash must remain unchanged.

Manual export example (use a fresh tag for a new experiment):

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/retrieve_items.py --dataset microlens --split val --method fusion --alpha 0.5 --gamma 0.8 --save-candidates --k 100 --manifest-tag fusion_a0p5_g0p8
```

The existing Qwen reranking script resolves the untagged baseline filenames. Tagged tuned manifests are separate experiments and will need an explicit input-path option before reranking; do not rename them over the baseline. Retrieval recall is a candidate-coverage upper bound, not measured Qwen reranking quality. Top-100 doubles candidate scoring calls relative to Top-50 in the per-candidate Qwen pipeline.
