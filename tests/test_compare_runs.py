import pandas as pd

from scripts.compare_runs import compare


def _run(tmp_path, name, rows):
    run = tmp_path / name
    run.mkdir()
    pd.DataFrame(rows, columns=["image_url", "face_bbox"]).to_parquet(run / "filtered_dataset.parquet")
    return run


def test_identical_finds_compare_equal(tmp_path):
    rows = [("u1", [0, 0, 120, 140]), ("u2", [5, 5, 130, 150])]
    assert compare(_run(tmp_path, "a", rows), _run(tmp_path, "b", rows)) == (set(), set())


def test_different_finds_are_listed(tmp_path):
    a = _run(tmp_path, "a", [("u1", [0, 0, 120, 140]), ("u2", [5, 5, 130, 150])])
    b = _run(tmp_path, "b", [("u1", [0, 0, 120, 140]), ("u3", [1, 1, 110, 120])])
    lost, gained = compare(a, b)
    assert lost == {("u2", (5, 5, 130, 150))}
    assert gained == {("u3", (1, 1, 110, 120))}
