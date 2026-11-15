"""Regression coverage for the new data, evaluation and resumable scoring paths."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch
import importlib.util

import numpy as np

from mmrr.artifacts import save_json, save_jsonl, sha256
from mmrr.data.schema import Item, Sequence
from mmrr.eval.experiment import evaluate_natural, fused_runs, paired_bootstrap
from mmrr.hars import neighbor_probabilities, preference_hardness
from mmrr.interest import retrieve_interest
from mmrr.io import read_jsonl
from mmrr.prepare import export_candidates, matched_training_rows, pointwise_examples, prepare_training
from mmrr.score import score_manifest


class PipelineTest(TestCase):
    @staticmethod
    def records():
        return [
            {"user": "hit", "history": [1], "target": 3, "candidates": [2, 3],
             "scores": [0.4, 0.3], "target_rank": 2},
            {"user": "miss", "history": [1], "target": 4, "candidates": [2, 3],
             "scores": [0.4, 0.3], "target_rank": None},
        ]

    def test_interest_preserves_exclusions_recent_weight_and_ties(self):
        ids = np.arange(1, 6)
        vectors = np.float32([[1, 0], [0, 1], [1, 0], [0, 1], [0, 1]])
        seq = Sequence("user", [1, 2], 5)
        _, result, scores = next(retrieve_interest([seq], ids, vectors, k=3))
        self.assertEqual(result, [4, 5, 3])
        self.assertEqual(scores, [1.0, 1.0, 0.75])

    def test_sample_training_pairs_match_and_exclude_old_observations(self):
        row = {"user": "u", "history": [2, 3, 4, 5, 6], "target": 7,
               "candidates": [7, 8, 1, 9, 10, 11]}
        opts = dict(seed=8, negatives=2, rank_low=1, rank_high=5)
        observations = {"u": list(range(1, 8))}
        inputs = ([row], observations, list(range(1, 14)))
        random, mined = next(matched_training_rows(*inputs, **opts))
        self.assertEqual((random, mined), next(matched_training_rows(*inputs, **opts)))
        self.assertTrue(all(x["item"] > 7 for r in (random, mined) for x in r["negatives"]))
        self.assertTrue(all(x["rank"] in (2, 4, 5) for x in mined["negatives"]))
        for r in (random, mined):
            points = list(pointwise_examples([r]))
            self.assertEqual([p["chosen"] for p in points], ["Yes", "No", "No"])
            self.assertEqual(points[0]["candidate"], 7)
            self.assertTrue(all(p["positive_item"] == 7 for p in points))

    def test_train_data_artifacts_round_trip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "microlens"
            source.mkdir()
            (source / "train.tsv").write_text("101\t1 2\n102\t3 4\n", encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            ids = np.arange(1, 16)
            vectors = np.eye(512, dtype=np.float32)[:len(ids)]
            for name, data in (("ids", ids), ("text", vectors), ("img", vectors),
                               ("has_img", np.ones(len(ids), dtype=bool))):
                np.save(cache / f"item_{name}.npy", data)
            manifest = root / "train.jsonl"
            output = root / "examples"
            with patch("mmrr.prepare.DATA_ROOT", root), patch("mmrr.data.loaders.paths.DATA_ROOT", root):
                self.assertEqual(export_candidates("microlens", "train", cache, manifest, 8, 2, None), 2)
                self.assertEqual(prepare_training("microlens", manifest, cache, output, 2025, 4, 1, 8), 2)
            self.assertEqual(len(list(read_jsonl(output / "random.jsonl"))), 10)
            self.assertEqual(len(list(read_jsonl(output / "retrieved.jsonl"))), 10)

    def test_full_denominator_and_fusion_endpoints(self):
        rows = self.records()
        scored = [{**rows[0], "scores": [0.2, 0.8], "reranked": [3, 2]}]
        result, ranks = evaluate_natural(rows, scored)
        self.assertEqual(result["candidate_recall"], 0.5)
        self.assertEqual(result["ranking"]["HR@3"], 0.5)
        self.assertEqual(result["conditional_ranking"]["HR@3"], 1)
        with self.assertRaisesRegex(ValueError, "Missing scores"):
            evaluate_natural(rows, [])
        baseline, before = evaluate_natural(rows, fused_runs(rows, scored, 0.0))
        _, after = evaluate_natural(rows, fused_runs(rows, scored, 1.0))
        self.assertEqual(baseline["ranking"], result["retrieval"])
        self.assertEqual(after, ranks)
        self.assertGreater(paired_bootstrap(before, after, samples=8)["NDCG@3"]["difference"], 0)
        with self.assertRaisesRegex(ValueError, "Invalid natural"):
            evaluate_natural([{**rows[0], "oracle_injected": True}])

    def test_hars_weight_uses_pair_distances(self):
        ids = [1, 2, 3, 4]
        vectors = np.float32([[1, 0], [1, 0], [0, 1], [0, -1]])
        probabilities = neighbor_probabilities(vectors, k=2, block_size=2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        weights = preference_hardness([{"positive_item": 1, "negative_item": 2},
                                       {"positive_item": 1, "negative_item": 3}], ids, probabilities)
        self.assertLess(weights[0], 1)
        self.assertGreater(weights[1], 1)

    def test_score_resumes_without_loading_model(self):
        class DummyScorer:
            def score_messages(self, messages):
                return [0.0, 1.0]

        def no_more_scoring(_):
            self.fail("Completed shards should not be rescored")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "val.jsonl"
            rows = self.records()
            save_jsonl(manifest, rows)
            save_json(manifest.with_suffix(".meta.json"), {
                "manifest_sha256": sha256(manifest), "dataset": "microlens", "split": "val",
                "n_users": 2, "k": 2,
            })
            args = SimpleNamespace(manifest=manifest, output=root / "run", dataset="microlens",
                                   hits_only=True, max_users=None, template="onerec", history_images=False,
                                   no_candidate_images=False, adapter=None, sft_adapter=None,
                                   model="dummy", revision="test", precision="fp32", batch_size=1,
                                   device="cpu", shard_users=1)
            catalog = {i: Item(i, f"item {i}", None) for i in range(1, 5)}
            result = score_manifest(args, lambda _: DummyScorer(), lambda _, **kwargs: catalog)
            self.assertEqual(len(list(read_jsonl(result))), 1)
            self.assertEqual(score_manifest(args, no_more_scoring), result)
            args.hits_only = False
            with self.assertRaisesRegex(ValueError, "different inputs"):
                score_manifest(args, no_more_scoring)


@skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
class PreferenceTest(TestCase):
    def test_answer_head_matches_full_projection_and_supports_repeated_backward(self):
        import torch

        from mmrr.models.preference import AnswerHead

        head = torch.nn.Linear(8, 4, bias=False)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        expected = head(hidden)[:, -1:, :]
        torch.testing.assert_close(AnswerHead(head, 1, 2, False)(hidden), expected)
        preference = AnswerHead(head, 1, 2, True)
        for _ in range(2):
            margin = preference(hidden).squeeze(-1)
            torch.testing.assert_close(margin, expected[:, :, 1] - expected[:, :, 2])
            margin.sum().backward()
        self.assertIsNone(head.weight.grad)

    def test_signed_margin_matches_one_token_dpo(self):
        import torch
        import torch.nn.functional as F

        from mmrr.models.preference import preference_loss

        policy = torch.tensor([[3.0, -2.0], [5.0, 1.0], [2.0, 0.0]], requires_grad=True)
        ref = torch.tensor([[1.0, 0.0], [2.0, 3.0], [0.0, 1.0]])
        yes = torch.tensor([True, False, True])
        sign = torch.where(yes, 1, -1)
        delta = policy.log_softmax(-1) - ref.log_softmax(-1)
        expected = -F.logsigmoid(0.1 * sign * (delta[:, 0] - delta[:, 1])).mean()
        loss = preference_loss(policy[:, 0] - policy[:, 1], ref[:, 0] - ref[:, 1], yes, 0.1)
        torch.testing.assert_close(loss, expected)
        weighted = preference_loss(policy[:, 0] - policy[:, 1], ref[:, 0] - ref[:, 1], yes, 0.1,
                                   torch.tensor([0.9, 1.1, 1.0]))
        weighted.backward()
        self.assertTrue(torch.isfinite(policy.grad).all())

    def test_nodo_survives_checkpoint_recomputation_and_removes_hooks(self):
        import torch
        from torch.utils.checkpoint import checkpoint

        from mmrr.models.preference import nodo_noise

        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.active_adapters = ["default"]
                self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})
                self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 2, bias=False)})

            def forward(self, x):
                return self.lora_B["default"](self.lora_A["default"](x))

        model = Adapter()
        x = torch.ones(1, 2, requires_grad=True)
        before = model(x).detach()
        gradients = []
        for call in (model, lambda tensor: checkpoint(model, tensor, use_reentrant=False)):
            model.zero_grad()
            x.grad = None
            torch.manual_seed(12)
            with nodo_noise(model, 0.05):
                call(x).square().sum().backward()
            gradients.append([parameter.grad.clone() for parameter in model.parameters()])
        for first, second in zip(*gradients, strict=True):
            torch.testing.assert_close(first, second)
        torch.testing.assert_close(model(x).detach(), before)
