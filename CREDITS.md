# Data provenance & credits

## WIT / Wikimedia

Qualitative crops under [`docs/latest_run_showcase/qualitative_analysis/`](docs/latest_run_showcase/qualitative_analysis/) (Final Targets, FN-candidate galleries, and the README thumbnails) derive from public [Wikipedia-based Image Text (WIT)](https://github.com/google-research-datasets/wit) / [Wikimedia Commons](https://commons.wikimedia.org/) images (`wikimedia/wit_base`).

Upstream WIT was **privacy-scrubbed** of many **primary-subject** faces. That sparse INPUT haystack is the constraint this pipeline ran against. The galleries show the **needles the pipeline found** — public Wikimedia-derived qualitative QA crops — not a claim that demo outputs are anonymized.

## Per-crop source pointers

Each gallery `metadata.csv` includes an `image_url` column (and the gallery HTML links the same URL) pointing at the Commons file page for that crop. Author, title, and license live on the Commons file page — this file does not restate a per-image license inventory.

- [`docs/latest_run_showcase/qualitative_analysis/final_targets/metadata.csv`](docs/latest_run_showcase/qualitative_analysis/final_targets/metadata.csv)
- [`docs/latest_run_showcase/qualitative_analysis/false_negative_candidates/metadata.csv`](docs/latest_run_showcase/qualitative_analysis/false_negative_candidates/metadata.csv)

## Dual note (software vs showcase media)

- **Software and project docs** in this repository are licensed under [PolyForm Noncommercial 1.0.0](LICENSE).
- **Showcase media** (face crops, gallery HTML that embeds them, and `image_url` targets) remain under their **upstream Commons / WIT terms** (often CC BY, CC BY-SA, or public domain). They are **not** claimed as PolyForm NC works.

## Reuse

For any specific image, follow the license and attribution on that file’s Wikimedia Commons page.
