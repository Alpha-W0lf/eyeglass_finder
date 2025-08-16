# Dev Guide: Two-Stage Worker Pool Implementation

This guide provides a sequential,-task-based checklist for refactoring the pipeline into a two-stage, process-isolated architecture.

## Phase 1: Refactor `src/processing/worker.py` for Specialized Roles

The goal of this phase is to split the monolithic worker functions into distinct, single-responsibility functions for detection and classification.

-   [x] **Task 1.1: Create Detection Worker.**
    -   [x] Rename `initialize_worker` to `initialize_detection_worker`.
    -   [x] Modify `initialize_detection_worker` to *only* load the face detector model (`g_face_detector`).
    -   [ ] Rename `process_chunk_of_data` to `run_detection_worker`.
    -   [ ] Modify `run_detection_worker` to only perform the detection stage logic (calling `detect_and_crop_faces`).
    -   [ ] Update `run_detection_worker`'s return value to be `None`. Success is communicated via the `_SUCCESS` file.
    -   [ ] Implement the "Atomic Success Marker": upon successful completion, `run_detection_worker` must create a `_SUCCESS` file in its output directory.

-   [ ] **Task 1.2: Create Classification Worker.**
    -   [ ] Create a new function `initialize_classification_worker` that *only* loads the glasses classifier models (`g_glasses_classifiers`).
    -   [ ] Create a new function `run_classification_worker` that accepts a `worker_data_dir` path as its primary argument.
    -   [ ] Implement the logic within `run_classification_worker` to:
        -   [ ] Scan the given directory for face metadata files.
        -   [ ] Read the data and call `classify_faces`.
        -   [ ] Write the classification and audit results back to the same directory.
        -   [ ] Update the function's return value to be `None`. Errors are communicated by raising exceptions.

## Phase 2: Implement the Two-Stage Worker Pool

The goal of this phase is to refactor the main pipeline to use a two-stage worker pool.

-   [x] **Task 2.1: Implement Detection Stage.**
    -   [x] Restructure the `process_images` function to clearly define the "Detection Stage".
    -   [x] Instantiate the first `ProcessPoolExecutor`, configured to use `initialize_detection_worker` and call `run_detection_worker`.
    -   [x] Adapt the existing `tqdm` progress bar for this stage.
    -   [x] Implement the "fail-fast" error handling logic for this pool.
    -   [x] Ensure the main process waits for all detection futures to complete before proceeding.

-   [x] **Task 2.2: Implement Data Discovery & Handoff.**
    -   [x] After the detection pool finishes, add logic to scan the `temp_worker_data` directory.
    -   [x] Create a list of all subdirectory paths that contain a `_SUCCESS` file. This list is the workload for the next stage.
-   [x] **Task 2.3: Implement Classification Stage.**
    -   [x] Clearly define the "Classification Stage" in `process_images`.
    -   [x] Instantiate a **new**, second `ProcessPoolExecutor`, configured to use `initialize_classification_worker` and call `run_classification_worker`.
    -   [x] Create a new `tqdm` progress bar for this stage.
    -   [x] Submit the list of directory paths from the discovery stage to this new pool.
    -   [x] Implement fail-fast logic for the classification pool.
    -   [x] Ensure the main process waits for all classification futures to complete.
-   [x] **Task 2.4: Finalize Orchestration.**
    -   [x] Verify that the final `_aggregate_worker_data` function is called correctly *after* the classification stage is complete.
    -   [x] Update all relevant logging (`PROGRESS_DASHBOARD`, etc.) to include stage-specific prefixes (`[Detection]`, `[Classification]`) for clarity.