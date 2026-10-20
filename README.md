# Eyeglass Finder

**A needle-in-a-haystack search: finds people wearing eyeglasses in Wikipedia images, inside a dataset whose publishers ran a face detector to screen out portraits. Tuned and moved to the Apple M2 Max GPU, it runs ~21× faster than the CPU baseline: 10.6 → 225 images/sec.**

| | | |
|:---:|:---:|:---:|
| ![Find 1](docs/latest_run_showcase/qualitative_analysis/final_targets/target_340_clf1.00_det0.92.jpg) | ![Find 2](docs/latest_run_showcase/qualitative_analysis/final_targets/target_266_clf1.00_det0.92.jpg) | ![Find 3](docs/latest_run_showcase/qualitative_analysis/final_targets/target_243_clf1.00_det0.91.jpg) |
| ![Find 4](docs/latest_run_showcase/qualitative_analysis/final_targets/target_64_clf1.00_det0.88.jpg) | ![Find 5](docs/latest_run_showcase/qualitative_analysis/final_targets/target_66_clf1.00_det0.88.jpg) | ![Find 6](docs/latest_run_showcase/qualitative_analysis/final_targets/target_286_clf1.00_det0.88.jpg) |

<sub>Six finds, each confirmed in a blind review. Face crops from Wikimedia Commons; credits in [`CREDITS.md`](CREDITS.md).</sub>

**Browse the 38 confirmed finds scored 0.97 or above:** [Finds gallery](https://alpha-w0lf.github.io/eyeglass_finder/docs/latest_run_showcase/qualitative_analysis/final_targets/index.html) · [Near misses below the cutoff](https://alpha-w0lf.github.io/eyeglass_finder/docs/latest_run_showcase/qualitative_analysis/false_negative_candidates/index.html)

## Results

The task: find people wearing eyeglasses (not sunglasses) whose face is at least 100×100 px. Showcase run `run_2026-10-19_01-20-39`:

| Metric | Value |
|---|---|
| Throughput | **225 images/sec**, ~21× the 10.6 images/sec baseline |
| Images scanned | 39,258 (2 WIT shards) in 174 s, 0 decode errors |
| Images with no face detected | 31,990 (81.5%) |
| Faces found / big enough to judge | 15,465 / 414 |
| **Finds (eyeglasses score ≥ 0.90)** | **44** |
| Precision, blind AI review | **43 of 44** ([labels](docs/latest_run_showcase/visual_review.csv)) |
| Recall, blind AI review | an estimated 74% (95% interval 56–86%) ([labels](docs/latest_run_showcase/visual_review_2.csv)) |

<sub>Both speed numbers come from committed run reports: the pre-tuning code in Docker on the CPU ([baseline](docs/baseline_run/report.md), 10.58) and the tuned code on the GPU ([showcase](docs/latest_run_showcase/report.md), 225.05); 225.05 / 10.58 = 21.27. How precision and recall were checked, with every label: [`docs/post_run_report.md`](docs/post_run_report.md#26-how-the-results-were-checked).</sub>

## The problem

Wikimedia's WIT image dataset is built to avoid portraits: its publishers ran the RetinaFace detector and left out images where a person is the primary subject, a face covering more than 10% of the picture ([dataset card](https://huggingface.co/datasets/wikimedia/wit_base)). In the showcase run, the face detector found no face in 81% of the 39,258 images. The pipeline has to find the faces that remain, keep those big enough to judge, and decide which ones wear eyeglasses, at speed.

## How it works

```mermaid
graph LR
    A[WIT parquet shards] --> B[Stream in chunks]
    B --> C[10 worker processes]
    C --> D[YOLOv8-Face: find faces]
    D --> E[Crop faces >= 100 px]
    E --> F[Eyeglasses classifier]
    F --> G[annotated_faces.parquet]
    G --> H[Filter + report + galleries]
```

Stage 1 (`scripts/process_data.py`) runs the models, saves every candidate face, then builds the report. Stage 2 (`scripts/generate_run_artifacts.py`) does the filtering and reporting and can be re-run alone, so a new threshold or plot needs no new model pass. Hardware is picked automatically: NVIDIA CUDA, then Apple MPS, then CPU (`src/utils/device.py`).

## Key engineering decisions

1. **Two models, two stages.** YOLOv8-Face detects the faces. A pre-trained eyeglasses classifier ([`glasses-detector`](https://github.com/mantasu/glasses-detector)) scores each face. Each model can be swapped or retuned on its own, and the expensive model pass is stored so filtering and reporting re-run without it.
2. **Native Apple GPU, tuned for throughput: ~21×.** I moved from CPU-only Docker to the native Apple GPU (MPS), batched face detection, scaled to 10 worker processes, and added memory-aware prefetching so long runs stay stable. A test on Apple-GPU Macs pins the GPU face boxes to the CPU result, so a framework upgrade can't silently change which faces are found.
3. **Every run reports on itself.** Each run writes a data funnel, timings, CPU and memory plots, and browsable galleries of the finds and the near misses, so every result can be reviewed visually.

## Try it

Needs Python 3.12 and Poetry. The face-detector weights download on first run and are checked against a pinned SHA-256.

```bash
poetry install
poetry run pytest -q        # unit tests, CPU only, no data needed
# Download train-00000-of-00330.parquet and train-00001-of-00330.parquet
# from https://huggingface.co/datasets/wikimedia/wit_base/tree/main/data into data/raw/
poetry run python scripts/process_data.py --config config/production.yaml
```

The run writes `outputs/run_<timestamp>/` with `report.md`, `filtered_dataset.parquet`, plots and galleries. Docker setup (portable, CPU only on macOS) and every config option: [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

**Stack:** Python 3.12 · PyTorch 2.6 · Ultralytics YOLOv8-Face · glasses-detector · pandas / Parquet · pytest + ruff in GitHub Actions · Docker · Poetry

## Deeper docs

| Path | What it holds |
|---|---|
| `docs/latest_run_showcase/` | The full report, plots and galleries from the showcase run |
| `docs/post_run_report.md` | How the pipeline went from 10.6 to 225 images/sec, and how the results were checked |
| `docs/TECHNICAL.md` | Setup, design choices, output schemas |
| `dataset_card.md` | Output dataset schema and statistics |
| `config/production.yaml` | The tuned config used for the showcase run |
| `src/`, `scripts/`, `tests/` | Pipeline code, Stage 1 and Stage 2 entry points, unit tests |
| `AGENTS.md` | Commands and rules for coding agents working in this repo |

**How this was built:** AI-native. Coding agents write code under my plans and review, following [`AGENTS.md`](AGENTS.md). Unit tests and lint run in CI on every pull request and every push to `main`.

**Credits and license:** The showcase face crops come from public Wikimedia Commons / WIT images; see [`CREDITS.md`](CREDITS.md). Code and docs: GNU AGPL-3.0 ([`LICENSE`](LICENSE)), the same licence as the Ultralytics library the pipeline runs on.

Built by Thomas Chacko, Senior AI Engineer (Ex-Meta). [LinkedIn](https://www.linkedin.com/in/tchacko1/)
