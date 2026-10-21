import json
from pathlib import Path
import runpy
import sys
import numpy as np
import pytest
from mmrr import paths
from mmrr.data import loaders
from mmrr.data.schema import Item
from mmrr.models import qwen_scorer


def test_explicit_candidate_file_and_output_protection(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'OUTPUT_ROOT', tmp_path/'outputs')
    monkeypatch.setattr(loaders, 'load_catalog', lambda _: {i: Item(i,str(i),None) for i in (1,2,3)})
    class FakeScorer:
        def score_messages(self,msgs):
            return np.array([0.,1.])
        def meta(self):
            return {'backend':'fake'}
    seen = {}
    def make(backend,**kw):
        seen.update(kw)
        return FakeScorer()
    monkeypatch.setattr(qwen_scorer, 'make_scorer', make)
    manifest = tmp_path/'tagged.jsonl'
    manifest.write_text(json.dumps({'user':'u','history':[1],'target':3,'candidates':[2,3],'scores':[1.,0.],'target_rank':2})+'\n')
    script = Path(__file__).resolve().parents[1]/'scripts/rerank_qwen.py'
    monkeypatch.setattr(sys,'argv',[str(script),'--candidates',str(manifest),'--k','2','--backend','hf','--hf-batch-size','1','--tag','test'])
    runpy.run_path(str(script),run_name='__main__')
    assert seen['batch_size'] == 1
    out = paths.OUTPUT_ROOT/'rerank/test/microlens/val.jsonl'
    assert json.loads(out.read_text())['reranked'] == [3,2]
    assert json.loads(out.with_suffix('.meta.json').read_text())['candidate_file']=='tagged.jsonl'
    original = out.read_bytes()
    with pytest.raises(SystemExit):
        runpy.run_path(str(script),run_name='__main__')
    assert out.read_bytes() == original


def test_metrics_keep_unretrieved_targets_in_denominator():
    script = Path(__file__).resolve().parents[1]/'scripts/evaluate_rerank.py'
    evaluate = runpy.run_path(str(script))['evaluate']
    rows = [{'user':'a','target':2,'candidates':[1,2],'reranked':[2,1]},
            {'user':'b','target':3,'candidates':[1,2],'reranked':[2,1]}]
    result = evaluate(rows)
    assert result['candidate_recall']==result['rerank_HR@3']==result['rerank_NDCG@3']==.5
