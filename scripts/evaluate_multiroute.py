"""Evaluate two fixed quota unions on all MicroLens validation queries."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
from mmrr.paths import DATA_ROOT, OUTPUT_ROOT, REPO_ROOT, emb_dir, candidates_path
from mmrr.data.loaders import load_sequences
from mmrr.retrieval import retrieve, item_vectors
from mmrr.multiroute import quota_union
from mmrr.eval.ranking import candidate_record, rank_of
from mmrr.io import read_jsonl, write_jsonl, write_meta


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = candidates_path('microlens', 'val', 50).parent
    configurations = [(30, 10, 10), (60, 20, 20)]
    paths = [root / f'val_multi_f{f}_t{t}_i{i}_a0p5_g0p5_top{f+t+i}.jsonl' for f,t,i in configurations]
    for p in paths:
        if p.exists() or p.with_suffix('.meta.json').exists():
            raise FileExistsError(f'Will not overwrite {p}')
    source = emb_dir('microlens')
    source_hash = {p.name: sha(p) for p in [*sorted(source.glob('*.npy')), DATA_ROOT / 'microlens/val.tsv']}
    ids = np.load(source / 'item_ids.npy')
    text = np.load(source / 'item_text.npy')
    image = np.load(source / 'item_img.npy')
    assert np.load(source / 'item_has_img.npy').all()
    assert np.isfinite(text).all() and np.isfinite(image).all()
    sequences = load_sequences('microlens', 'val')
    baseline_records, baseline_hashes = {}, {}
    for k in (50, 100):
        p = root / f'val_fusion_a0p5_g0p5_top{k}.jsonl'
        meta = json.loads(p.with_suffix('.meta.json').read_text(encoding='utf-8'))
        assert meta['source_sha256'] == source_hash
        assert meta['alpha'] == meta['gamma'] == .5
        assert sha(p) == meta['manifest_sha256']
        baseline_hashes[p] = sha(p)
        baseline_records[k] = list(read_jsonl(p))
        assert len(baseline_records[k]) == len(sequences)
    routes = []
    for method in ('fusion', 'text', 'image'):
        vectors = item_vectors(text, image, method, .5)
        routes.append([c for seq,c,s in tqdm(retrieve(sequences, ids, vectors=vectors, gamma=.5, k=100),
                                           total=len(sequences), desc=method)])
    results = []
    for quotas, path in zip(configurations, paths):
        k = sum(quotas)
        records, gains, losses, raw_sizes = [], 0, 0, []
        for index, seq in enumerate(sequences):
            query_routes = [r[index] for r in routes]
            c,s,origins = quota_union(query_routes, quotas)
            assert Counter(origins) == dict(enumerate(quotas))
            record = candidate_record(seq, c, s)
            record['candidate_sources'] = [('fusion','text','image')[o] for o in origins]
            records.append(record)
            base = baseline_records[k][index]
            assert (base['user'],base['history'],base['target']) == (seq.user,seq.history,seq.target)
            hit = seq.target in c
            base_hit = seq.target in base['candidates']
            gains += hit and not base_hit
            losses += base_hit and not hit
            raw_sizes.append(len(set().union(*(set(r[:q]) for r,q in zip(query_routes,quotas)))))
        write_jsonl(path, records)
        reread = list(read_jsonl(path))
        assert len(reread) == len(sequences)
        catalog = set(ids.tolist())
        for record in reread:
            c,s = record['candidates'],record['scores']
            assert len(c) == len(set(c)) == len(s) == k and set(c) <= catalog
            assert not set(c).intersection(record['history'])
            assert all(math.isfinite(v) for v in s) and s == sorted(s, reverse=True)
            assert record['target_rank'] == rank_of(c,record['target'])
        hits = sum(r['target_rank'] is not None for r in reread)
        baseline_hits = sum(r['target_rank'] is not None for r in baseline_records[k])
        assert hits-baseline_hits == gains-losses
        row = {'quotas': quotas, 'k': k, 'n_queries':len(records), 'hits':hits, 'recall':hits/len(records),
               'baseline_hits':baseline_hits, 'baseline_recall':baseline_hits/len(records),
               'gained_queries':gains,'lost_queries':losses,'net_hits':gains-losses,
               'mean_unique_before_backfill':float(np.mean(raw_sizes)), 'manifest':str(path), 'validation':'PASS'}
        write_meta(path.with_suffix('.meta.json'), {'dataset':'microlens','split':'val','alpha':.5,'gamma':.5,
                   'quotas':dict(zip(('fusion','text','image'),quotas)), 'k':k,'route_depth':100,
                   'selection':'fusion then text then image; skip duplicates and backfill each route quota',
                   'ordering':'equal-weight RRF c=60 on top100 route ranks; ordering only, no candidate selection',
                   'source_sha256':source_hash,'manifest_sha256':sha(path),
                   'code_sha256':{str(p.relative_to(REPO_ROOT)):sha(p) for p in [Path(__file__),REPO_ROOT/'src/mmrr/multiroute.py',REPO_ROOT/'src/mmrr/retrieval.py']},
                   'metrics':row})
        results.append(row)
    assert all(sha(p)==v for p,v in baseline_hashes.items())
    output = OUTPUT_ROOT / 'retrieval/microlens/val/multiroute_a0.5_g0.5'
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    main()
