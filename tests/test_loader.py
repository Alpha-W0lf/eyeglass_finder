import pandas as pd

from src.data_processing.loader import stream_data_generator


def _write_parquet(path, n):
    pd.DataFrame(
        {"image_url": [f"https://example.org/{i}.jpg" for i in range(n)], "image": [{"bytes": b"x"}] * n}
    ).to_parquet(path, index=False)


def test_chunks_respect_chunk_size_and_ids_run_across_files(tmp_path):
    first, second = tmp_path / "train-a.parquet", tmp_path / "train-b.parquet"
    _write_parquet(first, 5)
    _write_parquet(second, 3)

    chunks = list(stream_data_generator([first, second], chunk_size=2))

    assert [len(c) for c in chunks] == [2, 2, 1, 2, 1]
    ids = pd.concat(chunks)["image_id"].tolist()
    assert ids == list(range(8))
    assert chunks[0]["source_file"].iloc[0] == "train-a.parquet"
    assert chunks[-1]["source_file"].iloc[0] == "train-b.parquet"
