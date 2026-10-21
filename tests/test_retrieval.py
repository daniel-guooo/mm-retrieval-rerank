import math
import numpy as np
import pytest
from mmrr.data.schema import Sequence
from mmrr.retrieval import retrieve, summarize, popularity_scores, item_vectors


def test_retrieval_excludes_history_and_does_not_insert_target():
    ids = np.array([10, 20, 30, 40])
    vectors = np.array([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
    seqs = [Sequence('a', [10], 40), Sequence('b', [10], 20)]
    results = list(retrieve(seqs, ids, vectors=vectors, k=2))
    assert results[0][1] == results[1][1] == [20, 30]
    assert 10 not in results[0][1] and 40 not in results[0][1]


def test_recency_changes_rank_and_ties_use_id_order():
    ids = np.array([1, 2, 3, 4])
    vectors = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], dtype=np.float32)
    seqs = [Sequence('a', [1, 2], 3)]
    assert list(retrieve(seqs, ids, vectors=vectors, gamma=1, k=2))[0][1] == [3, 4]
    assert list(retrieve(seqs, ids, vectors=vectors, gamma=.5, k=2))[0][1] == [4, 3]


def test_popularity_counts_full_supplied_train_and_batching_matches():
    ids = np.array([1, 2, 3, 4])
    pop = popularity_scores(ids, [('train', [3, 3, 3, 1, 2])])
    assert pop.tolist() == [1, 1, 3, 0]
    seqs = [Sequence('a', [3], 4), Sequence('b', [1], 2)]
    a = list(retrieve(seqs, ids, popularity=pop, k=3, batch_size=1))
    b = list(retrieve(seqs, ids, popularity=pop, k=3, batch_size=2))
    assert a == b and a[0][1] == [1, 2, 4]


def test_metrics_include_misses_and_fusion_normalizes():
    m = summarize([1, 4, None], {1, 2}, 4)
    assert m['Recall@10'] == 2/3 and m['HR@3'] == 1/3
    assert m['NDCG@10'] == pytest.approx((1 + 1/math.log2(5))/3)
    assert m['coverage@50'] == .5
    fused = item_vectors(np.array([[1., 0.]]), np.array([[0., 1.]]), 'fusion')
    assert np.linalg.norm(fused) == pytest.approx(1)
    with pytest.raises(ValueError):
        list(retrieve([], np.array([1, 2]), popularity=np.ones(2), gamma=0))


@pytest.mark.parametrize('k,tag', [(50, ''), (100, ''), (50, 'tuned'), (100, 'tuned')])
def test_cli_saves_portable_full_split_manifest_and_refuses_overwrite(tmp_path, monkeypatch, k, tag):
    import json
    import runpy
    import sys
    from pathlib import Path
    from mmrr import paths
    monkeypatch.setattr(paths, 'DATA_ROOT', tmp_path / 'data')
    monkeypatch.setattr(paths, 'OUTPUT_ROOT', tmp_path / 'outputs')
    monkeypatch.setattr(paths, 'MANIFEST_DIR', tmp_path / 'manifests')
    data = paths.DATA_ROOT / 'microlens'
    data.mkdir(parents=True)
    (data / 'val.tsv').write_text('user1\t1 2\nuser2\t2 119\n', encoding='utf-8')
    emb = paths.emb_dir('microlens')
    emb.mkdir(parents=True)
    vectors = np.zeros((120, 512), dtype=np.float32)
    vectors[:, 0] = 1
    for name, value in {'item_ids': np.arange(1, 121), 'item_text': vectors,
                        'item_img': vectors, 'item_has_img': np.ones(120, dtype=bool)}.items():
        np.save(emb / f'{name}.npy', value)
    (emb / 'meta.json').write_text('{}', encoding='utf-8')
    script = str(Path(__file__).resolve().parents[1] / 'scripts/retrieve_items.py')
    monkeypatch.setattr(sys, 'argv', [script, '--method', 'fusion', '--save-candidates', '--k', str(k), '--manifest-tag', tag])
    runpy.run_path(script, run_name='__main__')
    manifest = paths.candidates_path('microlens', 'val', k)
    if tag:
        manifest = manifest.with_name(f'val_{tag}_top{k}.jsonl')
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 2 and all(len(r['candidates']) == k for r in records)
    assert records[0]['target_rank'] == 1 and records[1]['target_rank'] is None
    assert not any(set(r['history']) & set(r['candidates']) for r in records)
    assert manifest.with_suffix('.meta.json').exists()
    original = manifest.read_bytes()
    with pytest.raises(SystemExit):
        runpy.run_path(script, run_name='__main__')
    assert manifest.read_bytes() == original
