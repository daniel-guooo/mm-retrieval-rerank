from pathlib import Path
import random
import runpy

import torch

from mmrr.models.qwen_scorer import yes_no_margin

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/make_oracle_candidates.py'


def test_margin_matches_logit_difference_without_bf16_ties():
    torch.manual_seed(0)
    weight = torch.randn(10, 16).bfloat16()
    hidden = torch.randn(200, 16).bfloat16() * 8
    margin = hidden.float() @ yes_no_margin(weight, 3, 7)
    logits = hidden.float() @ weight.float().T
    assert torch.allclose(margin, logits[:, 3] - logits[:, 7], atol=1e-4)
    bf16 = (hidden @ weight.T).float()
    assert len(set((bf16[:, 3] - bf16[:, 7]).tolist())) < len(set(margin.tolist()))


def test_inject_drops_last_and_inserts_target_at_every_position():
    inject = runpy.run_path(str(SCRIPT))['inject']
    miss = {'user': 'a', 'history': [1], 'target': 9, 'candidates': [4, 5, 6], 'scores': [.3, .2, .1], 'target_rank': None}
    rng, ranks = random.Random(0), set()
    for _ in range(50):
        r = inject(miss, rng)
        assert [c for c in r['candidates'] if c != 9] == [4, 5] and 'scores' not in r
        assert r['target_rank'] == r['candidates'].index(9) + 1
        ranks.add(r['target_rank'])
    assert ranks == {1, 2, 3}


def test_split_modes_partition_users(tmp_path, monkeypatch):
    import json, sys
    src = tmp_path/'c.jsonl'
    rows = [{'user': 'a', 'history': [1], 'target': 9, 'candidates': [4, 5], 'scores': [.2, .1], 'target_rank': None},
            {'user': 'b', 'history': [1], 'target': 5, 'candidates': [4, 5], 'scores': [.2, .1], 'target_rank': 2}]
    src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for mode, user in (('hits', 'b'), ('inject', 'a')):
        monkeypatch.setattr(sys, 'argv', [str(SCRIPT), str(src), '--mode', mode])
        runpy.run_path(str(SCRIPT), run_name='__main__')
        out = [json.loads(l) for l in (tmp_path/f'c_{mode}.jsonl').read_text().splitlines()]
        assert [r['user'] for r in out] == [user]
    assert json.loads((tmp_path/'c_hits.jsonl').read_text()) == rows[1]
