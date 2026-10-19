import pandas as pd

from scripts.generate_run_artifacts import generate_final_dataset


def test_final_dataset_keeps_only_targets(tmp_path):
    df = pd.DataFrame(
        {
            "image_url": ["u0", "u1", "u2"],
            "source_file": ["f", "f", "f"],
            "face_bbox": [[0, 0, 10, 10]] * 3,
            "face_score": [0.9, 0.8, 0.95],
            "cropped_face_jpeg": [b"a", b"b", b"c"],
            "is_target": [True, False, True],
        }
    )

    generate_final_dataset(df, tmp_path)

    out = pd.read_parquet(tmp_path / "filtered_dataset.parquet")
    assert out["image_url"].tolist() == ["u0", "u2"]
    assert list(out.columns) == ["image_url", "source_file", "face_bbox", "face_confidence", "face_jpeg"]


def test_no_targets_writes_no_file(tmp_path):
    df = pd.DataFrame(
        {
            "image_url": ["u0"],
            "source_file": ["f"],
            "face_bbox": [[0, 0, 10, 10]],
            "face_score": [0.9],
            "cropped_face_jpeg": [b"a"],
            "is_target": [False],
        }
    )
    generate_final_dataset(df, tmp_path)
    assert not (tmp_path / "filtered_dataset.parquet").exists()
