"""Compare seven history representations at a fixed Top-50 budget, validation only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
from mmrr.paths import DATA_ROOT, OUTPUT_ROOT, REPO_ROOT, emb_dir, candidates_path
from mmrr.data.loaders import load_sequences
from mmrr.retrieval import item_vectors
from mmrr.interest import METHODS, interest_scores
from mmrr.eval.ranking import candidate_record
from mmrr.io import read_jsonl,write_jsonl,write_meta


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description='Compare history representations at Top-50 or Top-100')
    ap.add_argument('--k', type=int, choices=(50,100), default=50)
    k = ap.parse_args().k
    metric = f'Recall@{k}'
    output = OUTPUT_ROOT/f'retrieval/microlens/val/history_representations_a0.5_top{k}'
    if output.exists():
        raise FileExistsError(output)
    source = emb_dir('microlens')
    hashes = {p.name:sha(p) for p in [*sorted(source.glob('*.npy')),DATA_ROOT/'microlens/val.tsv']}
    baseline_path = candidates_path('microlens','val',k).with_name(f'val_fusion_a0p5_g0p5_top{k}.jsonl')
    baseline_hash = sha(baseline_path)
    meta = json.loads(baseline_path.with_suffix('.meta.json').read_text(encoding='utf-8'))
    assert meta['source_sha256']==hashes and meta['manifest_sha256']==baseline_hash
    assert meta['alpha']==meta['gamma']==.5
    baseline = list(read_jsonl(baseline_path))
    seqs = load_sequences('microlens','val')
    assert len(seqs)==len(baseline)==5000
    ids=np.load(source/'item_ids.npy')
    assert np.all(ids[1:]>ids[:-1]) and np.load(source/'item_has_img.npy').all()
    vectors=item_vectors(np.load(source/'item_text.npy'),np.load(source/'item_img.npy'),'fusion',.5)
    assert np.isfinite(vectors).all()
    lookup={int(x):i for i,x in enumerate(ids)}
    records={method:[] for method in METHODS}
    for start in tqdm(range(0,len(seqs),128),desc=f'All 7 methods; Top-{k}'):
        chunk=seqs[start:start+128]
        histories=[[lookup[x] for x in seq.history] for seq in chunk]
        flat=[x for h in histories for x in h]
        sim=vectors[flat]@vectors.T
        offset=0
        for j,(seq,h) in enumerate(zip(chunk,histories)):
            base=baseline[start+j]
            assert (seq.user,seq.history,seq.target)==(base['user'],base['history'],base['target'])
            per_history=sim[offset:offset+len(h)]
            offset+=len(h)
            for method in METHODS:
                scores=interest_scores(per_history,method)
                scores[h]=-np.inf
                order=np.argsort(-scores,kind='stable')[:k]
                records[method].append(candidate_record(seq,ids[order].tolist(),scores[order].tolist()))
    all_records={'baseline_gamma05':baseline,**records}
    rows=[]
    base_hits=[r['target_rank'] is not None for r in baseline]
    for method,recs in all_records.items():
        hits=[r['target_rank'] is not None for r in recs]
        ranks=[r['target_rank'] for r in recs]
        rows.append({'method':method,'hits':sum(hits),metric:sum(hits)/len(recs),
                     'HR@10':sum(r is not None and r<=10 for r in ranks)/len(recs),
                     'NDCG@10':sum(1/math.log2(r+1) for r in ranks if r is not None and r<=10)/len(recs),
                     'gained':sum(a and not b for a,b in zip(hits,base_hits)),
                     'lost':sum(b and not a for a,b in zip(hits,base_hits))})
    # Fixed method order resolves equal Recall@50; no test-set selection.
    best=max(rows,key=lambda r:r[metric])
    best_alternative=max(rows[1:],key=lambda r:r[metric])
    method=best_alternative['method']
    manifest=baseline_path.with_name(f'val_interest_{method}_a0p5_top{k}.jsonl')
    if manifest.exists() or manifest.with_suffix('.meta.json').exists():
        raise FileExistsError(manifest)
    write_jsonl(manifest,records[method])
    saved=list(read_jsonl(manifest))
    for r in saved:
        assert len(r['candidates'])==len(set(r['candidates']))==k
        assert not set(r['candidates']).intersection(r['history'])
        assert all(math.isfinite(s) for s in r['scores']) and r['scores']==sorted(r['scores'],reverse=True)
        rank=r['candidates'].index(r['target'])+1 if r['target'] in r['candidates'] else None
        assert rank==r['target_rank']
    assert len(saved)==5000 and sum(r['target_rank'] is not None for r in saved)==best_alternative['hits']
    assert sha(baseline_path)==baseline_hash
    prefix_checks = []
    if k == 100:
        previous = OUTPUT_ROOT/'retrieval/microlens/val/history_representations_a0.5_top50/comparison.json'
        if previous.exists():
            prior = json.loads(previous.read_text(encoding='utf-8'))
            for row in prior['results']:
                current = all_records[row['method']]
                assert sum(r['target_rank'] is not None and r['target_rank']<=50 for r in current)==row['hits']
            old_path = Path(prior['saved_manifest'])
            for old, new in zip(read_jsonl(old_path), records[prior['best_alternative']['method']]):
                assert old['user']==new['user'] and old['candidates']==new['candidates'][:50] and old['scores']==new['scores'][:50]
            prefix_checks.append('All 8 Top-50 hit counts reproduced; saved Top-50 candidates and scores match prefixes')
    config={'dataset':'microlens','split':'val','alpha':.5,'k':k,'method':method,'n_users':5000,
            'selection':f'Highest validation {metric} among seven predefined alternatives; fixed method order for ties',
            'source_sha256':hashes,'manifest_sha256':sha(manifest),
            'code_sha256':{p.name:sha(p) for p in [Path(__file__),REPO_ROOT/'src/mmrr/interest.py']},
            'scores':'per-candidate aggregation of cosine similarities to history items',
            'history_exclusion':'all input history','tie_break':'ascending item ID'}
    write_meta(manifest.with_suffix('.meta.json'),config)
    output.mkdir(parents=True)
    result={'results':rows,'best_overall':best,'best_alternative':best_alternative,'saved_manifest':str(manifest),
            'validation':'PASS','baseline_unchanged':True,'test_split_used':False,'candidate_budget':k,'prefix_checks':prefix_checks}
    (output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
