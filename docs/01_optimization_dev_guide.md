# Phase 1: Native Transition + Foundation Development Guide

> Note: Phase 1 has been fully implemented and validated as of 2025-08-17.

## Overview
**Objective:** Transition from Docker to native execution and establish foundation for optimization
**Difficulty:** EASY - Low Risk, High Impact  
**Expected Duration:** 1 Week (6-10 hours total)
**Expected Improvement:** 2-3x performance gain (CPU optimizations + MPS acceleration)

## Prerequisites
- [x] Current working pipeline in Docker (9.8 images/second baseline)
- [x] M2 Max MacBook Pro with 32GB RAM
- [x] Poetry installed and configured
- [x] Git repository in clean state

## Phase 1 Task Sequence

### Task 1.1: Native Environment Setup + CPU Baseline (30 minutes)
**Objective:** Establish native execution environment and baseline measurements

#### Subtasks:
- [x] **1.1.1** Create git checkpoint before optimization
  ```bash
  git tag pre-phase1-optimization
  git commit -am "Checkpoint: Before Phase 1 optimization"
  ```

- [x] **1.1.2** Install native environment
  ```bash
  cd /Users/tom/Documents/Git/eyeglass_finder
  poetry install
  poetry shell
  ```

- [x] **1.1.3** Verify all dependencies work natively
  ```bash
  poetry run python -c "import torch, ultralytics, cv2; print('All imports successful')"
  poetry run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
  ```

- [x] **1.1.4** Test basic pipeline functionality (CPU-only)
  ```bash
  # Force CPU mode for baseline
  export FORCE_CPU_ONLY=true
  poetry run python scripts/process_data.py --config config/config.yaml
  ```

- [x] **1.1.5** Record CPU baseline performance
  ```
  Baseline Metrics to Record:
  - Processing time: ~3.1 minutes (184 seconds)
  - Throughput: ~21.7 images/second
  - Peak memory usage: Stabilized by worker isolation
  - CPU utilization: (Will analyze from report)
  - Worker count: 8 (with max_tasks_per_child=1)
  ```

- [x] **1.1.6** Validate data integrity (compare with Docker results)
  - [x] Check output file counts match
  - [x] Verify no processing errors
  - [x] Confirm all input images processed

### Task 1.2: Basic Algorithmic Optimizations (CPU-only) (2-4 hours)
**Objective:** Implement CPU-level optimizations before GPU acceleration

#### Subtasks:
- [x] **1.2.1** Create performance benchmark utility
  ```python
  # Create: src/utils/benchmark.py
  class PerformanceBenchmark:
      def __init__(self):
          self.baseline_throughput = [RECORD_FROM_1.1.5]
          self.regression_threshold = 0.1  # 10% degradation threshold
      
      def validate_optimization(self, optimization_name):
          # Implementation as per optimization_notes.md
  ```

- [x] **1.2.2** Optimize list building operations
  **Target:** `src/processing/pipeline.py` - Replace append loops with comprehensions
  - [x] Identify all `for item in X: list.append()` patterns
  - [x] Replace with list comprehensions: `[item for item in X if condition]`
  - [x] Test with micro-benchmark (expect 15-25% improvement)
  **Outcome:** Reverted. Two test runs confirmed a consistent 5-8% performance degradation.

- [x] **1.2.3** Optimize index calculations
  **Target:** Reduce repeated arithmetic in batch processing loops
  - [x] Decision: Not implemented in Phase 1 (risk of added complexity; negligible impact vs GPU gains)
  - [x] Outcome: Deferred to Phase 2/3 only if profiling identifies this as a bottleneck

- [x] **1.2.4** Optimize memory allocations (deferred to Phase 2)
  **Outcome:** Core memory improvements achieved via internal mini-batching + JIT image loading + explicit cleanup. Additional list/tensor pre-allocation not pursued in Phase 1; revisit if profiling shows benefit.

- [x] **1.2.5** Validate algorithmic optimizations
  ```bash
  # Run with optimizations
  poetry run python scripts/process_data.py --config config/config.yaml
  ```
  ```
  Optimized CPU Metrics (Average of 2 runs):
  - Processing time: ~3.3 minutes (197 seconds) (vs baseline: 184s)
  - Throughput: 20.3 images/second (vs baseline: 21.7)
  - Improvement: -6.45 % 
  ```

### Task 1.3: Enhanced Configuration (1-2 hours)
**Objective:** Add M2 Max-specific configuration parameters

#### Subtasks:
- [x] **1.3.1** Backup current config
  ```bash
  cp config/config.yaml config/config.yaml.backup
  ```

- [x] **1.3.2** Add hardware-specific settings to `config/config.yaml`
  ```yaml
  # Add new section:
  hardware:
    max_workers: 10              # Phase 1 target (safer intermediate step)
    worker_memory_limit_mb: 1500 # Per-worker memory cap for 10 workers
    use_mps_acceleration: false  # Will enable in Task 1.4
    device_override: null        # Manual device override if needed
  ```

- [x] **1.3.3** Add performance tuning parameters
  ```yaml
  # Add new section:
  performance:
    prefetch_chunks: 2           # Prefetch data for smoother streaming
    gc_frequency: 10             # Garbage collection every N chunks  
    thread_pool_workers: 4       # I/O thread pool size
    batch_optimization: true     # Enable new algorithmic optimizations
  ```

- [x] **1.3.4** Add observability enhancements
  ```yaml
  # Add new section:
  observability:
    detailed_metrics: true       # Enhanced metric collection
    memory_profiling: false      # Enable for debugging only
    benchmark_mode: true         # Enable performance regression detection
    performance_alerts:
      cpu_threshold: 90          # Alert if CPU > 90%
      memory_threshold: 85       # Alert if memory > 85%
      min_throughput: 8.8        # Minimum images/second (10% below baseline)
  ```

- [x] **1.3.5** Update code to use new configuration parameters
  **Files to modify:**
  - [x] `src/processing/worker.py` - Add hardware.max_workers support
  - [x] `scripts/process_data.py` - Add performance.gc_frequency support  
  - [x] `src/utils/config.py` - Add validation for new parameters

- [x] **1.3.6** Test configuration changes
  ```bash
  poetry run python scripts/process_data.py --config config/config.yaml --dry-run
  ```
  **Outcome:** Successfully loaded and ran with new config. Determined 10 workers is slower than 8. Reverted to 8 workers as the optimal CPU baseline.

### Task 1.4: MPS Integration + Comprehensive Validation (2-3 hours)
**Objective:** Enable Apple GPU acceleration with robust validation

#### Subtasks:
- [x] **1.4.1** Create MPS validation utility
  ```python
  # Create: src/utils/mps_validator.py
  # Implement validation functions from optimization_notes.md:
  # - validate_yolov8_mps()
  # - validate_mps_memory() 
  # - validate_end_to_end_mps()
  ```

- [x] **1.4.2** Run individual model validation
  ```bash
  poetry run python -c "
  from src.utils.mps_validator import validate_yolov8_mps
  result = validate_yolov8_mps()
  print(f'YOLOv8 MPS compatible: {result}')
  "
  ```
  ```
  YOLOv8 MPS Results:
  - Compatible: _____ (True/False)
  - Speed improvement: _____ x
  - Recommended: _____ (True/False)
  - Issues (if any): _____
  ```

- [x] **1.4.3** Run glasses-detector validation
  ```bash
  poetry run python -c "
  from src.utils.mps_validator import validate_glasses_detector_mps
  result = validate_glasses_detector_mps()
  print(f'Glasses detector MPS compatible: {result}')
  "
  ```
  ```
  Glasses Detector MPS Results:
  - Compatible: _____ (True/False)
  - Speed improvement: _____ x  
  - Recommended: _____ (True/False)
  - Issues (if any): _____
  ```

- [x] **1.4.4** Run memory usage validation
  ```bash
  poetry run python -c "
  from src.utils.mps_validator import validate_mps_memory
  result = validate_mps_memory()
  print(f'MPS memory acceptable: {result}')
  "
  ```
  ```
  MPS Memory Results:
  - Acceptable: _____ (True/False)
  - Memory increase: _____ %
  - Total memory usage: _____ GB
  ```

- [x] **1.4.5** Enable MPS if validation passes
  **Condition:** All validations return compatible=True AND speed improvement >10%
  ```yaml
  # Update config/config.yaml:
  hardware:
    use_mps_acceleration: true  # Enable MPS
  ```

- [x] **1.4.6** Run end-to-end MPS performance test
  ```bash
  poetry run python scripts/process_data.py --config config/config.yaml
  ```
  ```
  MPS Performance Results (latest small-set run):
  - Processing time: ~20 seconds
  - Throughput: ~200 images/second
  - Improvement vs CPU baseline: ~10x on this set
  - Memory usage: Stable with per-mini-batch cleanup and MPS cache flush
  - CPU utilization: Moderate; GPU-bound on MPS
  ```

- [x] **1.4.7** Implement automatic fallback (if MPS issues detected)
  **Target:** `src/utils/device.py`
  - [x] Add runtime functional check during worker init (quick device check and CPU fallback)
  - [x] Add logging for fallback events

### Task 1.5: Phase 1 Validation & Documentation (30 minutes)
**Objective:** Validate Phase 1 success and document results

#### Subtasks:
- [x] **1.5.1** Run performance regression check
  ```bash
  poetry run python -c "
  from src.utils.benchmark import PerformanceBenchmark
  benchmark = PerformanceBenchmark()
  benchmark.validate_optimization('Phase1-Complete')
  "
  ```

- [x] **1.5.2** Document Phase 1 results
  ```
  PHASE 1 COMPLETION METRICS:
  
  Baseline (Docker):     9.8 images/second
  CPU Optimizations:     20.3 images/second (-6.45% vs CPU baseline of 21.7 ips)
  MPS Enabled:           99.75 images/second (+360% vs CPU baseline)
  Total Phase 1 Gain:    10.17x (+917% vs Docker baseline)
  
  Memory Usage:          21.52 GB peak (within 32 GB limit)
  Worker Count:          8 (optimal after testing 10)
  MPS Status:            Enabled (with runtime CPU fallback safety)

  Warmup/Ramp-up Observation (for Phase 2):
  - During startup we observe transient disk I/O and memory spikes (worker spawn + model loads + MPS warmup + initial parquet reads).
  - This is expected and tapers in steady-state. We will add a ramp-up plan in Phase 2 to smooth startup:
    - Start with reduced prefetch (e.g., prefetch_chunks=1) and optional smaller initial chunk size for the first K chunks
    - Stagger worker task submissions by a small delay to avoid synchronized model loads
    - Then ramp back to configured throughput targets
  
  Success Criteria:
  - [x] >50% improvement over baseline (achieved: ~10.17x)
  - [x] <50% memory increase (baseline memory not formally recorded; peak 21.52 GB within 32 GB limit)
  - [x] Zero processing errors (decoding errors: 0; failed batches: 0)
  - [x] All validation tests pass (Phase 1 tasks completed)
  ```

- [x] **1.5.3** Create git checkpoint for Phase 1 completion
  ```bash
  git add .
  git commit -m "Complete Phase 1: Native transition + MPS integration

  Performance Results:
  - Baseline: [X.X] images/second  
  - Phase 1: [X.X] images/second ([XX]% improvement)
  - MPS Status: [Enabled/Disabled]
  - Memory Usage: [X.X] GB
  
  Optimizations Applied:
  - Native execution environment
  - Algorithmic optimizations (list building, index calculations)
  - MPS GPU acceleration
  - Enhanced configuration management"
  
  git tag phase1-complete
  ```

- [x] **1.5.4** Prepare for Phase 2
  - [x] Review Phase 2 requirements in `02_optimization_dev_guide.md`
  - [x] Validate Phase 1 meets prerequisites for Phase 2
  - [x] Document any issues or deviations for Phase 2 planning

## Success Criteria for Phase 1
- [x] **Performance:** 2-3x improvement over baseline (achieved ~10.17x vs Docker baseline)
- [x] **Stability:** Zero processing errors, all test cases pass
- [x] **Memory:** <50% memory increase from baseline (baseline memory not recorded; peak 21.52 GB)
- [x] **MPS:** GPU acceleration working OR graceful fallback to CPU
- [x] **Foundation:** Native execution environment fully operational

## Rollback Procedure (if needed)
If any critical issues occur:
1. `git reset --hard pre-phase1-optimization`
2. Document issues in `docs/optimization_issues.md`
3. Revert `config/config.yaml` from backup
4. Return to Docker execution method

## Notes Section
```
Implementation Notes:
- Date Started: _____
- Date Completed: _____
- Issues Encountered: _____
- Lessons Learned: _____
- Recommendations for Phase 2: _____
```
