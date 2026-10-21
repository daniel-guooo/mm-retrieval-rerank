"""Launch a named frozen validation experiment from configs/qwen_handoff.json."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from mmrr.paths import REPO_ROOT


def main():
    cfg = json.loads((REPO_ROOT/'configs/qwen_handoff.json').read_text(encoding='utf-8'))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, choices=cfg['experiments'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--max-users', type=int)
    ap.add_argument('--hf-batch-size', type=int, default=1)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    ex = cfg['experiments'][args.experiment]
    if hashlib.sha256((REPO_ROOT/ex['manifest']).read_bytes()).hexdigest() != ex['sha256']:
        raise ValueError('Candidate manifest checksum mismatch')
    cmd = [sys.executable,'-X','utf8',str(REPO_ROOT/'scripts/rerank_qwen.py'),
           '--dataset','microlens','--split','val','--backend','hf','--template','onerec',
           '--k',str(ex['k']),'--candidates',str(REPO_ROOT/ex['manifest']),
           '--model',args.model,'--hf-batch-size',str(args.hf_batch_size),
           '--users-per-call','1','--seed','2025','--tag',args.tag]
    if args.max_users is not None:
        cmd += ['--max-users',str(args.max_users)]
    print(json.dumps(cmd,ensure_ascii=False),flush=True)
    if not args.dry_run:
        subprocess.run(cmd,check=True)


if __name__ == '__main__':
    main()
