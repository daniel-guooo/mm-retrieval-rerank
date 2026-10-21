import pytest

from mmrr.data.prompt_dataset import RerankPromptDataset, render_text
from mmrr.data.schema import Item, Sequence, validate_catalog, validate_sequences
from mmrr.models.qwen_prompt import (TEMPLATES, PromptConfig, build_messages, max_images_per_prompt,
                                     template_id, yes_no_token_ids)

CAT = {i: Item(i, f"title {i}", f"microlens/images/{i}.jpg") for i in range(1, 10)}
CAT[9] = Item(9, 'has "quotes"', None)  # 缺图


# ---------- schema ----------
def test_valid_catalog_passes_and_counts_missing_images():
    rep = validate_catalog(CAT, check_files=False)
    assert rep.ok, rep.errors
    assert rep.stats["missing_images"] == 1


def test_catalog_rejects_absolute_or_windows_paths_and_str_ids():
    bad = {1: Item(1, "a", "/Users/zz/x.jpg"), 2: Item(2, "b", "imgs\\2.jpg"), "3": Item("3", "c", "c.jpg")}
    assert len(validate_catalog(bad, check_files=False).errors) == 3


def test_sequences_validation():
    ok = [Sequence("u1", [1, 2, 3, 4, 5], 6), Sequence("u2", [1, 2], 3)]
    rep = validate_sequences(ok, CAT)
    assert rep.ok and rep.stats["history_len_dist"] == {2: 1, 5: 1}
    errs = validate_sequences([Sequence("u1", [1, 2, 3, 4, 5, 6], 7), Sequence("u1", [1], 99)], CAT).errors
    assert any("出现 2 次" in e for e in errs)
    assert any("长度 6" in e for e in errs)
    assert any("99" in e for e in errs)


# ---------- templates ----------
def _kinds(msgs):
    return [c["type"] for c in msgs[-1]["content"]]


def _text(msgs):
    return render_text(msgs)


HIST = [CAT[i] for i in range(1, 6)]


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_basic_shape(name):
    m = build_messages(HIST, CAT[7], PromptConfig(template=name))
    assert m[-1]["role"] == "user"
    assert _kinds(m).count("image") == 1
    t = _text(m)
    # 历史按时间顺序出现，候选在历史之后，结尾问 Yes/No
    assert t.index("title 1") < t.index("title 5") < t.index("title 7")
    assert "Yes" in t and "No" in t
    assert "video" in t and "movie" not in t


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_title_comes_right_before_its_image(name):
    m = build_messages(HIST, CAT[7], PromptConfig(template=name, history_images=True))
    parts = m[-1]["content"]
    assert _kinds(m).count("image") == 6 == max_images_per_prompt(5, PromptConfig(history_images=True))
    for i, p in enumerate(parts):
        if p["type"] == "image":
            item_id = int(p["image"].rsplit("/", 1)[1].split(".")[0])
            assert parts[i - 1]["text"].rstrip().endswith(f'"title {item_id}"')


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_prompts_of_same_user_share_prefix_up_to_candidate(name):
    cfg = PromptConfig(template=name, history_images=True)
    a, b = _text(build_messages(HIST, CAT[6], cfg)), _text(build_messages(HIST, CAT[7], cfg))
    common = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    assert "title 5" in a[:common] and "title 6" not in a[:common]


def test_missing_image_falls_back_to_title_only_and_quotes_are_escaped():
    m = build_messages(HIST, CAT[9], PromptConfig())
    assert _kinds(m).count("image") == 0
    assert "\"has 'quotes'\"" in _text(m)


def test_category_follows_dataset_and_unknown_template_errors():
    assert "movie" in _text(build_messages(HIST, CAT[7], PromptConfig(), dataset="netflix"))
    with pytest.raises(KeyError):
        build_messages(HIST, CAT[7], PromptConfig(template="nope"))
    assert template_id("onerec").startswith("onerec@")


class FakeTok:
    def __init__(self, table):
        self.table = table

    def encode(self, w, add_special_tokens=False):
        return self.table[w]


def test_yes_no_token_ids():
    assert yes_no_token_ids(FakeTok({"Yes": [9454], "No": [2753]})) == (9454, 2753)
    with pytest.raises(ValueError):
        yes_no_token_ids(FakeTok({"Yes": [1, 2], "No": [3]}))


# ---------- prompt dataset ----------
RECS = [
    {"user": "u1", "history": [1, 2, 3, 4, 5], "target": 7, "candidates": [6, 7, 8], "scores": [3, 2, 1], "target_rank": 2},
    {"user": "u2", "history": [2, 3], "target": 1, "candidates": [8, 9], "scores": [2, 1], "target_rank": None},
]


def test_dataset_flattens_user_candidate_pairs_in_order():
    ds = RerankPromptDataset(RECS, CAT)
    assert len(ds) == 5 and ds.n_users == 2
    got = [(e.user, e.candidate, e.position, e.label) for e in (ds[i] for i in range(len(ds)))]
    assert got == [("u1", 6, 0, 0), ("u1", 7, 1, 1), ("u1", 8, 2, 0), ("u2", 8, 0, 0), ("u2", 9, 1, 0)]
    with pytest.raises(IndexError):
        ds[5]


def test_dataset_user_batches_match_flat_order():
    ds = RerankPromptDataset(RECS, CAT)
    batches = list(ds.iter_user_batches(1))
    assert [len(r) for r, _ in batches] == [1, 1]
    flat = [e for _, exs in ds.iter_user_batches(5) for e in exs]
    assert [e.candidate for e in flat] == [ds[i].candidate for i in range(len(ds))]
    assert flat[0].image_paths[0].endswith("microlens/images/6.jpg")


def test_dataset_rejects_unknown_items():
    with pytest.raises(KeyError):
        RerankPromptDataset([{**RECS[0], "candidates": [404]}], CAT)


def test_dataset_meta():
    m = RerankPromptDataset(RECS, CAT, PromptConfig(template="minionerec")).meta()
    assert m["prompt"].startswith("minionerec@") and m["n_prompts"] == 5
