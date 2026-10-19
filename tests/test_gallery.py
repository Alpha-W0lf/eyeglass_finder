from scripts.generate_run_artifacts import (
    final_targets_header,
    near_miss_header,
    render_gallery_html,
)


def _record(image_id, proba):
    return {
        "filename": f"target_{image_id}_clf{proba:.2f}_det0.90.jpg",
        "image_id": image_id,
        "eyeglasses_proba": proba,
        "detector_conf": 0.9,
        "is_target": True,
        "sunglasses_prediction": False,
        "face_size": [120, 140],
        "face_bbox": [0, 0, 120, 140],
        "image_url": "https://upload.wikimedia.org/x.jpg",
    }


def test_finds_page_has_title_viewport_repo_link_and_alt_text():
    records = [_record(1, 0.99), _record(2, 0.98)]
    title, header = final_targets_header("run_x", total=2, shown=2, threshold=0.9, gallery_min=0.0)
    html = render_gallery_html(records, title, header)

    assert "<title>Eyeglass Finder: the 2 finds</title>" in html
    assert 'name="viewport"' in html
    assert 'href="https://github.com/Alpha-W0lf/eyeglass_finder"' in html
    assert html.count("alt='Face crop, image_id") == 2
    assert html.count("class='card'") == 2


def test_finds_page_says_when_it_shows_only_the_clearest():
    title, header = final_targets_header("run_x", total=125, shown=92, threshold=0.9, gallery_min=0.97)
    assert title == "Eyeglass Finder: the 92 clearest finds"
    assert "kept 125 faces" in header
    assert "scored 0.97 or higher" in header


def test_near_miss_header_states_score_range_and_cutoff():
    title, header = near_miss_header(count=20, low=0.832, high=0.899, threshold=0.9)
    assert title == "Eyeglass Finder: near misses"
    assert "scores 0.83 to 0.90, cutoff 0.90" in header


def test_generator_shows_only_clear_finds_but_keeps_every_find(tmp_path):
    from pathlib import Path

    import pandas as pd

    from scripts.generate_run_artifacts import create_qualitative_samples
    from src.utils.config import load_config

    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    config.report_generation.gallery_visual_review_csv = None
    config.report_generation.gallery_min_eyeglasses_proba = 0.97
    probs = [0.99, 0.98, 0.92, 0.85, 0.40]
    df = pd.DataFrame(
        {
            "is_target": [p >= 0.9 for p in probs],
            "has_eyewear_confidence": probs,
            "face_score": [0.9] * 5,
            "cropped_face_jpeg": [b"jpeg"] * 5,
            "image_url": [f"https://upload.wikimedia.org/{i}.jpg" for i in range(5)],
            "face_size": [[120, 140]] * 5,
            "face_bbox": [[0, 0, 120, 140]] * 5,
            "sunglasses_prediction": [False] * 5,
        }
    )

    create_qualitative_samples(df, tmp_path, config)

    finds = tmp_path / "qualitative_analysis" / "final_targets"
    html = (finds / "index.html").read_text()
    assert html.count("class='card'") == 2
    assert "the 2 clearest finds" in html
    assert len(pd.read_csv(finds / "metadata.csv")) == 3
    assert len(list(finds.glob("*.jpg"))) == 3
    near = (tmp_path / "qualitative_analysis" / "false_negative_candidates" / "index.html").read_text()
    assert near.count("class='card'") == 2


def test_generator_shows_every_find_when_none_reach_the_gallery_minimum(tmp_path):
    from pathlib import Path

    import pandas as pd

    from scripts.generate_run_artifacts import create_qualitative_samples
    from src.utils.config import load_config

    config = load_config(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    assert config.report_generation.gallery_min_eyeglasses_proba == 0.0
    config.report_generation.gallery_min_eyeglasses_proba = 0.99
    df = pd.DataFrame(
        {
            "is_target": [True, True],
            "has_eyewear_confidence": [0.95, 0.93],
            "face_score": [0.9, 0.9],
            "cropped_face_jpeg": [b"jpeg", b"jpeg"],
            "image_url": ["https://upload.wikimedia.org/a.jpg", "https://upload.wikimedia.org/b.jpg"],
            "face_size": [[120, 140]] * 2,
            "face_bbox": [[0, 0, 120, 140]] * 2,
            "sunglasses_prediction": [False, False],
        }
    )

    create_qualitative_samples(df, tmp_path, config)

    html = (tmp_path / "qualitative_analysis" / "final_targets" / "index.html").read_text()
    assert html.count("class='card'") == 2
    assert "This page shows all 2" in html


def test_default_header_for_other_galleries():
    html = render_gallery_html([], "Eyeglass Finder: rejected samples", "<header>\n<h1>Eyeglass Finder: rejected samples</h1>\n</header>")
    assert "<title>Eyeglass Finder: rejected samples</title>" in html


def test_face_size_from_parquet_arrays_renders_as_a_plain_list(tmp_path):
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from scripts.generate_run_artifacts import create_qualitative_samples
    from src.utils.config import load_config

    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    config.report_generation.gallery_visual_review_csv = None
    df = pd.DataFrame(
        {
            "is_target": [True],
            "has_eyewear_confidence": [0.99],
            "face_score": [0.9],
            "cropped_face_jpeg": [b"jpeg"],
            "image_url": ["https://upload.wikimedia.org/a.jpg"],
            "face_size": [np.array([114, 164])],
            "face_bbox": [np.array([95, 49, 209, 214])],
            "sunglasses_prediction": [False],
        }
    )
    create_qualitative_samples(df, tmp_path, config)
    html = (tmp_path / "qualitative_analysis" / "final_targets" / "index.html").read_text()
    assert "<b>face_size:</b> [114, 164]" in html
    assert "<b>face_bbox:</b> [95, 49, 209, 214]" in html


def test_cards_show_the_source_image_id_and_files_use_the_face_row(tmp_path):
    from pathlib import Path

    import pandas as pd

    from scripts.generate_run_artifacts import create_qualitative_samples
    from src.utils.config import load_config

    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    config.report_generation.gallery_visual_review_csv = None
    df = pd.DataFrame(
        {
            "image_id": [5012, 5012],
            "is_target": [True, True],
            "has_eyewear_confidence": [0.99, 0.98],
            "face_score": [0.9, 0.9],
            "cropped_face_jpeg": [b"a", b"b"],
            "image_url": ["https://upload.wikimedia.org/a.jpg"] * 2,
            "face_size": [[120, 140]] * 2,
            "face_bbox": [[0, 0, 120, 140], [200, 0, 320, 140]],
            "sunglasses_prediction": [False, False],
        }
    )
    create_qualitative_samples(df, tmp_path, config)
    finds = tmp_path / "qualitative_analysis" / "final_targets"
    html = (finds / "index.html").read_text()
    assert html.count("<b>image_id:</b> 5012") == 2
    assert sorted(p.name for p in finds.glob("*.jpg")) == ["target_0_clf0.99_det0.90.jpg", "target_1_clf0.98_det0.90.jpg"]
    assert "src='target_0_clf0.99_det0.90.jpg'" in html


def test_finds_page_shows_only_finds_confirmed_in_visual_review(tmp_path):
    from pathlib import Path

    import pandas as pd

    from scripts.generate_run_artifacts import create_qualitative_samples
    from src.utils.config import load_config

    visual_review = tmp_path / "confirmed_visual_review.csv"
    visual_review.write_text(
        "image_id,face_index,glasses_confidence,visual_review\n"
        "1,0,0.9900,eyeglasses\n2,0,0.9800,none\n3,0,0.9500,eyeglasses\n"
    )
    config = load_config(Path(__file__).resolve().parent.parent / "config" / "production.yaml")
    config.report_generation.gallery_visual_review_csv = str(visual_review)
    df = pd.DataFrame(
        {
            "image_id": [1, 2, 3],
            "face_index": [0, 0, 0],
            "is_target": [True, True, True],
            "has_eyewear_confidence": [0.99, 0.98, 0.95],
            "face_score": [0.9] * 3,
            "cropped_face_jpeg": [b"a", b"b", b"c"],
            "image_url": [f"https://upload.wikimedia.org/{i}.jpg" for i in range(3)],
            "face_size": [[120, 140]] * 3,
            "face_bbox": [[0, 0, 120, 140]] * 3,
            "sunglasses_prediction": [False] * 3,
        }
    )

    create_qualitative_samples(df, tmp_path, config)

    finds = tmp_path / "qualitative_analysis" / "final_targets"
    html = (finds / "index.html").read_text()
    assert "<title>Eyeglass Finder: finds confirmed in a blind visual review</title>" in html
    assert "flagged 3 faces" in html
    assert "a blind visual review confirmed 2" in html
    assert "shows the 1 confirmed finds scored 0.97 or above" in html
    assert html.count("class='card'") == 1
    assert "<b>image_id:</b> 1<" in html
    assert len(pd.read_csv(finds / "metadata.csv")) == 3


def test_production_config_points_at_the_committed_visual_review():
    from pathlib import Path

    from src.utils.config import load_config

    root = Path(__file__).resolve().parent.parent
    path = load_config(root / "config" / "production.yaml").report_generation.gallery_visual_review_csv
    assert (root / path).exists()


def test_near_miss_header_can_say_a_face_is_left_out():
    _, header = near_miss_header(count=20, low=0.60, high=0.89, threshold=0.9, left_out=1)
    assert "These are the 20 highest-scoring faces" in header
    assert "1 of them is left out of this page because the source image is not on Wikimedia Commons." in header


def test_cards_link_the_commons_file_page():
    from scripts.generate_run_artifacts import commons_file_page

    url = "https://upload.wikimedia.org/wikipedia/commons/7/7f/Toni_%26_Eva.jpg"
    assert commons_file_page(url) == "https://commons.wikimedia.org/wiki/File:Toni_%26_Eva.jpg"
    assert commons_file_page("https://example.org/x.jpg") == "https://example.org/x.jpg"
    html = render_gallery_html([_record(1, 0.99)], "t", "<header></header>")
    assert "href='https://upload.wikimedia.org/x.jpg'" in html  # non-Commons URLs pass through
