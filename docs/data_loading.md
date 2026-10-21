# Data loading and audit

## Accepted policy

- Keep official splits, original integer IDs and original row/interaction order.
- Keep sequences of length 2 or greater. History is the last up to five events before the final target. Length 0/1 sequences are excluded and their users are listed in the audit.
- Missing/empty titles use `[unknown ID]`; HTML entities, Unicode NFC and whitespace are normalized without truncation. Duplicate titles retain separate IDs. Literal `NaN` strings and replacement characters are reported for review, not guessed or silently repaired.
- Missing, empty or undecodable images have `image_path=None`. Valid images keep their original files/modes and portable paths relative to `MMRR_DATA_ROOT`.
- The catalog includes the union of title IDs, numeric image IDs and IDs from all three official splits. Split labels are not used for fitting; this union prevents silently dropping items absent from the title table.
- `read_interactions(dataset, split)` returns all source events. Use its **train** output for popularity; `load_sequences` is intentionally truncated. It does not authorize using validation/test interactions for fitting.
- TSVs have no timestamps. Order is preserved, but oldest-to-newest chronology remains a release assumption and cannot be independently verified from these files.
- Duplicate users, duplicate title IDs, malformed rows and ambiguous image files fail loudly rather than silently deduplicating.

## Run on Windows PowerShell

Set these variables in each new terminal, using the paths on your machine:

```powershell
$env:MMRR_DATA_ROOT = '<extracted archive>/data'
$env:MMRR_OUTPUT_ROOT = '<output directory>'
python -m pytest -q
python scripts/check_loader.py --dataset microlens
python scripts/check_loader.py --dataset movielens
python scripts/check_loader.py --dataset netflix
python scripts/audit_data.py --tokenizer '<local CLIP model directory>' --archive '<data-mllmrec.zip>'
```

The data root must directly contain `microlens`, `movielens`, `netflix`. Run from the repository directory after `pip install -e .`.

For data-only work, install `requirements-data.txt`. Full model inference still requires the project's model dependencies and a separately validated GPU setup. This delivery was checked with Python 3.12; Python 3.10 compatibility is not yet tested.

`audit_data.py` reads raw files without modifying them. It emits per-dataset JSON with affected IDs/users and source TSV/title hashes, a Markdown summary, and an optional archive SHA256. Token length is measured with the local CLIP tokenizer including special tokens; if no tokenizer is supplied, that check is explicitly marked null rather than zero.

## Repository repair

The initial checkout was missing the entire `src/mmrr/data` package because the unanchored `data/` ignore rule also excluded source directories. The rule now excludes only `/data/`. Schema validation and the lazy prompt adapter were reconstructed from the tracked callers and tests; they are not claimed to be recovered upstream implementations.

## Separate integration issue

The GPU setup follow-up corrects `ClipEncoder.MAX_TEXT_LEN` from 500 to 77. Data loaders still preserve complete titles; truncation happens only during CLIP encoding. `scripts/check_gpu_clip.py` checks real image/text features and a synthetic overlength title on CUDA.
