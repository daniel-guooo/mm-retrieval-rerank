"""Validation-only fusion grid and verified Top-50/100 exports. Never overwrites manifests."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from mmrr.paths import OUTPUT_ROOT, REPO_ROOT, candidates_path
from mmrr.data.loaders import load_sequences
from mmrr.io import read_jsonl
from mmrr.eval.ranking import rank_of


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(alpha, gamma, extra=()):
    subprocess.run([sys.executable, '-X', 'utf8', str(REPO_ROOT / 'scripts/retrieve_items.py'),
                    '--dataset', 'microlens', '--split', 'val', '--method', 'fusion',
                    '--alpha', str(alpha), '--gamma', str(gamma), *extra], check=True)


def main():
    output = OUTPUT_ROOT / 'retrieval/microlens/val/grid'
    output.mkdir(parents=True, exist_ok=True)
    old = candidates_path('microlens', 'val', 50)
    original_hash = sha(old)
    rows = []
    for alpha in (.25, .5, .75):
        for gamma in (.5, .8, 1.):
            run(alpha, gamma)
            path = output.parent / f'fusion_a{alpha:g}_g{gamma:g}' / 'metrics.json'
            rows.append({'alpha': alpha, 'gamma': gamma, **json.loads(path.read_text())[0]})
    # Prespecified: primary objective Recall@50; secondary cutoff resolves ties.
    # Remaining ties prefer less recency decay, then a balanced fusion weight.
    def key(row, k):
        return (row[f'Recall@{k}'], row[f'Recall@{150-k}'], row['gamma'], -abs(row['alpha']-.5))
    best50 = max(rows, key=lambda row: key(row, 50))
    best100 = max(rows, key=lambda row: key(row, 100))
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = {'dataset': 'microlens', 'split': 'val', 'grid': rows,
                 'primary_objective': 'Recall@50', 'tie_break': 'other cutoff, gamma descending, alpha closest to 0.5',
                 'best_at_50': best50, 'best_at_100': best100,
                 'scope': 'Best among the 9 evaluated configurations; test split untouched'}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    baseline = next(r for r in rows if r['alpha'] == .5 and r['gamma'] == 1.)
    configs = [('', baseline)]
    seen = {(.5, 1.)}
    for best in (best50, best100):
        pair = (best['alpha'], best['gamma'])
        if pair not in seen:
            tag = f"fusion_a{pair[0]:g}_g{pair[1]:g}".replace('.', 'p')
            configs.append((tag, best))
            seen.add(pair)
    sequences = load_sequences('microlens', 'val')
    audits = []
    for tag, row in configs:
        records_by_k = {}
        for k in (50, 100):
            path = candidates_path('microlens', 'val', k)
            if tag:
                path = path.with_name(f'val_{tag}_top{k}.jsonl')
            if not path.exists():
                extra = ['--save-candidates', '--k', str(k)]
                if tag:
                    extra += ['--manifest-tag', tag]
                run(row['alpha'], row['gamma'], extra)
            meta = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
            assert meta['alpha'] == row['alpha'] and meta['gamma'] == row['gamma'] and meta['k'] == k
            assert meta['manifest_sha256'] == sha(path)
            current_meta = json.loads((output.parent / f"fusion_a{row['alpha']:g}_g{row['gamma']:g}" / 'meta.json').read_text(encoding='utf-8'))
            assert meta['source_sha256'] == current_meta['source_sha256']
            assert meta['retrieval_code_sha256'] == current_meta['retrieval_code_sha256']
            records = list(read_jsonl(path))
            assert len(records) == len(sequences)
            hits = 0
            for r, seq in zip(records, sequences):
                assert (r['user'], r['history'], r['target']) == (seq.user, seq.history, seq.target)
                c, s = r['candidates'], r['scores']
                assert len(c) == len(set(c)) == len(s) == k
                assert not set(c).intersection(seq.history)
                assert all(math.isfinite(v) for v in s) and all(a >= b for a, b in zip(s, s[1:]))
                assert r['target_rank'] == rank_of(c, seq.target)
                hits += r['target_rank'] is not None
            assert math.isclose(hits / len(records), row[f'Recall@{k}'])
            records_by_k[k] = records
            audits.append({'path': str(path), 'k': k, 'alpha': row['alpha'], 'gamma': row['gamma'],
                           'queries': len(records), 'hits': hits, 'recall': hits/len(records), 'sha256': sha(path), 'status': 'PASS'})
        for r50, r100 in zip(records_by_k[50], records_by_k[100]):
            assert r50['candidates'] == r100['candidates'][:50]
            assert r50['scores'] == r100['scores'][:50]
    assert sha(old) == original_hash
    report = {'status': 'PASS', 'original_top50_unchanged': True, 'prefix_consistency': True, 'files': audits}
    (output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'best50': best50, 'best100': best100, 'validation': report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
