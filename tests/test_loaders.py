import csv
import pytest
from PIL import Image
from mmrr import paths
from mmrr.data.loaders import load_catalog, load_sequences, read_interactions


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    folder = tmp_path / "microlens"
    (folder / "images").mkdir(parents=True)
    with (folder / "Microlens_titles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "title"])
        writer.writerows([(1, "  A &amp; B\n"), (2, ""), (3, "same"), (4, "same"), (8, "catalog only")])
    Image.new("L", (8, 8)).save(folder / "images/1.jpg")
    (folder / "images/2.jpg").write_bytes(b"invalid image")
    (folder / "images/3.jpg").write_bytes(b"")
    Image.new("RGB", (32, 32)).save(folder / "images/9.jpg")
    (folder / "train.tsv").write_text("u1\t1 2\nu2\t1 2 3 4 5 6 7\nu3\t1\nu4\t1 1\n", encoding="utf-8")
    (folder / "val.tsv").write_text("v1\t1 10\n", encoding="utf-8")
    (folder / "test.tsv").write_text("t1\t1 3\n", encoding="utf-8")
    return folder


def test_catalog_union_and_missing_modalities(release):
    catalog = load_catalog("microlens")
    assert list(catalog) == list(range(1, 11))
    assert catalog[1].title == "A & B"
    assert catalog[1].image_path == "microlens/images/1.jpg"
    assert catalog[2].title == "[unknown 2]"
    assert catalog[10].title == "[unknown 10]"
    assert catalog[2].image_path is None and catalog[3].image_path is None
    assert catalog[3].title == catalog[4].title
    assert catalog[9].image_path == "microlens/images/9.jpg"
    assert load_catalog("microlens") == catalog


def test_short_sequences_order_repeats_and_full_train(release):
    seqs = load_sequences("microlens", "train")
    assert [s.user for s in seqs] == ["u1", "u2", "u4"]
    assert seqs[0].history == [1] and seqs[0].target == 2
    assert seqs[1].history == [2, 3, 4, 5, 6] and seqs[1].target == 7
    assert seqs[2].target in seqs[2].history
    assert read_interactions("microlens", "train")[1][1] == list(range(1, 8))
    assert seqs == load_sequences("microlens", "train")


def test_duplicate_users_and_invalid_arguments_fail(release):
    with (release / "train.tsv").open("a") as f:
        f.write("u1\t1 2\n")
    with pytest.raises(ValueError, match="duplicate user"):
        load_sequences("microlens", "train")
    with pytest.raises(ValueError):
        load_sequences("microlens", "../test")
    with pytest.raises(ValueError):
        load_catalog("../microlens")


def test_no_stale_root_cache(release, tmp_path, monkeypatch):
    load_catalog("microlens")
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "different")
    with pytest.raises(FileNotFoundError):
        load_catalog("microlens")


def test_audit_accounts_for_exclusions_and_cold_targets(release):
    import runpy
    from pathlib import Path
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_data.py"))["audit"]
    report = audit("microlens", None)
    assert report["splits"]["train"]["raw_sequences"] == 4
    assert report["splits"]["train"]["sequences"] == 3
    assert report["splits"]["train"]["excluded_under_2_users"] == ["u3"]
    assert report["splits"]["train"]["target_in_history"] == 1
    assert report["splits"]["val"]["cold_target_count"] == 1
    assert report["images"]["status_counts"] == {"ok": 2, "unreadable": 1, "zero_bytes": 1, "missing": 6}
    assert report["titles"]["duplicate_title_groups"] == 1
    assert report["titles"]["over_77_tokens_count"] is None
    assert report["table1"]["interactions"] == 16
