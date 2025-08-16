## Pipeline Enhancements Plan (Accuracy, Observability, Configurability)

### Status from latest full run
- Pipeline completed successfully; no critical errors. Prior warnings have been addressed (Ultralytics config dir, integrity accounting). Resource monitoring enabled; artifacts written.

### Accuracy: eyeglasses vs sunglasses misclassification
- Problem: Some images removed for “sunglasses” actually show eyeglasses. Likely causes:
  - Over-aggressive sunglasses decision threshold.
  - Lack of a tie-break rule when both eyeglasses and sunglasses scores are moderately high.
  - Single-face-only flow may miss additional faces with eyeglasses if `keep_all` is false.

- Proposals:
  1) Decision thresholds and tie-breaks (configurable):
     - `model_params.classification.eyeglasses_threshold: float` (default 0.50)
     - `model_params.classification.sunglasses_threshold: float` (default 0.50)
     - `model_params.classification.keep_if_eyeglasses_margin_over_sunglasses: float` (default 0.10)
     - `model_params.classification.decision_rule: [eyeglasses_overrides, sunglasses_overrides, margin]`
       where `margin` means keep if eyeglasses_score ≥ sunglasses_score + margin.

  2) Improve recall for multi-face images:
     - Set `model_params.face_detection.keep_all: true` and update pipeline to keep if any face satisfies eyeglasses criteria.
     - Add `model_params.face_detection.iou_threshold: float` and `max_det: int` to tune NMS and cap detections.

  3) Calibration & validation:
     - Build a small labeled validation set; compute PR/ROC, choose thresholds via Youden’s J or balanced F1.
     - Optionally fit Platt scaling/temperature for better score calibration.

### Observability: richer per-decision insight
- Current: Aggregates, samples, resource utilization, and integrity checks.
- Gaps: Limited visibility into negatives (images removed due to no-face or no‑eyeglasses) and sunglasses decision rationale.

- Proposals:
  1) Per-image decision audit (Parquet): one row per face with fields:
     - `image_id, face_id, detector_conf, bbox, face_size, eyeglasses_score, sunglasses_score, decision, reason`.
     - For images with no faces: 1 row with `reason = "no_face"`.
     - Store alongside cropped face path when available for quick triage.

  2) Negative sampling bundles:
     - Save sampled images for each removal reason: `no_face`, `below_min_face_size`, `sunglasses_only`, `low_eyeglasses_score`, `quality_issues`.
     - Include an index CSV mapping file → reason → scores.

  3) Report enhancements:
     - Add a “Decision Breakdown” section with counts per reason, top uncertain cases (|eyeglasses_score − sunglasses_score| < margin), and trend charts across files.
     - If a labeled set is present: confusion-like summary, ROC/PR curves, threshold sweep table.

### Configurability additions (for tuning)
- `model_params.face_detection`:
  - `keep_all: bool` (already present; default true for recall)
  - `min_confidence: float` (existing)
  - `iou_threshold: float` (new; default 0.5)
  - `max_det: int` (new; default 100)

- `model_params.classification`:
  - `eyeglasses_threshold: float` (new; default 0.50)
  - `sunglasses_threshold: float` (new; default 0.50)
  - `keep_if_eyeglasses_margin_over_sunglasses: float` (new; default 0.10)
  - `decision_rule: str` in {`margin`, `eyeglasses_overrides`, `sunglasses_overrides`} (new; default `margin`)

- `data_processing`:
  - `diagnostic_sampling_rate: float` (new; default 0.05) for saving negative examples.

### Ensuring we capture all qualifying eyeglasses
- Turn on `keep_all` and accept if any detected face meets eyeglasses criteria.
- Consider reducing `min_face_size` slightly (e.g., 100 → 80) with a compensating higher `eyeglasses_threshold`, then evaluate.
- Add a small retry path: if no faces found but the image likely contains a face (heuristic via image stats), run detector once more with a slightly lower confidence.

### Evaluation workflow
- Add an `evaluation/` script to load the per-image audit Parquet and (optionally) labeled CSV to compute metrics and produce threshold-sweep plots.
- Persist evaluation artifacts under each run directory for comparability.

### Performance considerations
- Increase `execution.num_workers` and `inference_batch_size` cautiously, guided by `resource_utilization.json`.
- Add `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_MAX_THREADS=1`, `TORCH_NUM_THREADS=1` to avoid thread over‑subscription when scaling workers.
- Consider prefetching next chunk while current chunk is processed (bounded queue) once correctness and observability are solid.

### Additional domains to demonstrate seniority
- Reproducibility: seeds, environment capture, artifact manifests, deterministic preprocessing where possible.
- Testing: unit tests for decision rules; integration tests for end‑to‑end runs on a small fixture dataset; regression tests for thresholds.
- Failure handling: robust retries with backoff for transient I/O; graceful degradation when a model file is missing with clear guidance.
- Security and PII: ensure no sensitive data persists in logs; scrub EXIF if storing images outside build context.

### Next actions (incremental)
1) Add config keys listed above and wire them through pipeline decision logic.
2) Implement per-image decision audit Parquet and negative sampling bundles.
3) Enhance the run report with decision breakdown and uncertainty analysis.
4) Build a small labeled validation set and add an evaluation script with threshold sweep.
5) Iterate thresholds using the labeled set, then lock defaults in `config.yaml`.
