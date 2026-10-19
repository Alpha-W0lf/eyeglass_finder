import hashlib

import pytest

from src.modeling import weights


def _pin(monkeypatch, name, content):
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(weights, "MODEL_WEIGHTS", {name: ("https://example.org/" + name, digest)})


def _fake_download(content, calls):
    def fetch(url, dest):
        calls.append(url)
        with open(dest, "wb") as f:
            f.write(content)
    return fetch


def test_missing_weights_are_downloaded_and_checked(tmp_path, monkeypatch):
    calls = []
    _pin(monkeypatch, "face.pt", b"weights")
    monkeypatch.setattr(weights.urllib.request, "urlretrieve", _fake_download(b"weights", calls))

    path = weights.ensure_weights(tmp_path / "models" / "face.pt")

    assert path.read_bytes() == b"weights"
    assert calls == ["https://example.org/face.pt"]


def test_download_with_wrong_bytes_is_rejected_and_not_kept(tmp_path, monkeypatch):
    _pin(monkeypatch, "face.pt", b"weights")
    monkeypatch.setattr(weights.urllib.request, "urlretrieve", _fake_download(b"tampered", []))

    with pytest.raises(ValueError):
        weights.ensure_weights(tmp_path / "face.pt")
    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_checked_without_downloading(tmp_path, monkeypatch):
    calls = []
    _pin(monkeypatch, "face.pt", b"weights")
    monkeypatch.setattr(weights.urllib.request, "urlretrieve", _fake_download(b"weights", calls))
    (tmp_path / "face.pt").write_bytes(b"weights")
    assert weights.ensure_weights(tmp_path / "face.pt") == tmp_path / "face.pt"
    assert calls == []

    (tmp_path / "face.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        weights.ensure_weights(tmp_path / "face.pt")


def test_production_model_has_a_pinned_download():
    from pathlib import Path

    from src.utils.config import load_config

    root = Path(__file__).resolve().parent.parent
    model_path = load_config(root / "config" / "production.yaml").model_params.face_detection.model_path
    assert Path(model_path).name in weights.MODEL_WEIGHTS
