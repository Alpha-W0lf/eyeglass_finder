# Credits and licences

## Showcase images (Wikimedia Commons, via WIT)

The face crops in [`docs/latest_run_showcase/qualitative_analysis/`](docs/latest_run_showcase/qualitative_analysis/) and the README thumbnails are **cropped and resized** from Wikimedia Commons images in the [Wikipedia-based Image Text (WIT)](https://github.com/google-research-datasets/wit) dataset (`wikimedia/wit_base`).

- [`docs/latest_run_showcase/image_credits.csv`](docs/latest_run_showcase/image_credits.csv) lists every committed crop with its Commons file page, author and licence (read from Commons on 2026-10-19). All are public domain or Creative Commons (CC BY or CC BY-SA, versions 2.0 to 4.0).
- Each gallery card links the same Commons file page. Follow the licence and attribution on that page when you reuse an image.
- The full, uncropped photos the pipeline saves for review (images with the most faces) are not included in this repository.

These images keep their own Commons licences. They are not covered by this repository's licence.

## Third-party code

- **glasses-detector** by Mantas Birškus, vendored in [`libs/glasses-detector/`](libs/glasses-detector/): MIT licence ([LICENSE](libs/glasses-detector/LICENSE), [upstream](https://github.com/mantasu/glasses-detector)). Its classifier weights are downloaded at runtime from the project's GitHub releases. Its documentation (upstream `docs/`, published at https://mantasu.github.io/glasses-detector/) and example images (`data/demo/`) are not included in this copy; see the upstream repository.

## Third-party models

Model weights are not committed. [`src/modeling/weights.py`](src/modeling/weights.py) downloads each file from its source on first use and checks a pinned SHA-256.

| File | Source | Licence |
|---|---|---|
| `yolov8n-face-lindevs.pt` | [lindevs/yolov8-face release 1.0.1](https://github.com/lindevs/yolov8-face/releases/tag/1.0.1), trained by Lindevs on WIDER FACE from Ultralytics YOLOv8 weights | Repository: [MIT](https://github.com/lindevs/yolov8-face/blob/master/LICENSE). Ultralytics YOLOv8, which the model builds on and which the pipeline runs it with, is [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |

The Python dependencies, including `ultralytics` (AGPL-3.0), are installed from PyPI under their own licences; see `pyproject.toml`.

## This repository

Code and docs: [GNU AGPL-3.0](LICENSE), copyright Thomas Chacko. See [`NOTICE`](NOTICE).
