"""Evaluate all users, retaining misses outside the candidate set as zero."""
import argparse
import json
import math
from pathlib import Path
from mmrr.io import read_jsonl
from mmrr.eval.ranking import rank_of


def evaluate(records):
    if not records or len({r['user'] for r in records}) != len(records):
        raise ValueError('Empty results or duplicate users')
    for r in records:
        if len(r['candidates']) != len(set(r['candidates'])) or sorted(r['candidates']) != sorted(r['reranked']):
            raise ValueError('Reranked items must be a permutation of unique candidates')
    result = {'n_users': len(records)}
    for label,key in [('retrieval','candidates'),('rerank','reranked')]:
        ranks = [rank_of(r[key],r['target']) for r in records]
        for k in (3,10):
            result[f'{label}_HR@{k}'] = sum(r is not None and r<=k for r in ranks)/len(ranks)
            result[f'{label}_NDCG@{k}'] = sum(1/math.log2(r+1) for r in ranks if r is not None and r<=k)/len(ranks)
    result['candidate_recall'] = sum(r['target'] in r['candidates'] for r in records)/len(records)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path)
    args = ap.parse_args()
    result = evaluate(list(read_jsonl(args.input)))
    output = args.input.with_suffix('.metrics.json')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
