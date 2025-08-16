# Phase 1: Native Transition + Foundation Development Guide

## Overview
**Objective:** Transition from Docker to native execution and establish foundation for optimization
**Difficulty:** EASY - Low Risk, High Impact  
**Expected Duration:** 1 Week (6-10 hours total)
**Expected Improvement:** 2-3x performance gain (CPU optimizations + MPS acceleration)

## Prerequisites
- [ ] Current working pipeline in Docker (9.8 images/second baseline)
- [ ] M2 Max MacBook Pro with 32GB RAM
- [ ] Poetry installed and configured
- [ ] Git repository in clean state

## Phase 1 Task Sequence

### Task 1.1: Native Environment Setup + CPU Baseline (30 minutes)
**Objective:** Establish native execution environment and baseline measurements

#### Subtasks:
- [ ] **1.1.1** Create git checkpoint before optimization
  ```bash
  git tag pre-phase1-optimization
  git commit -am "Checkpoint: Before Phase 1 optimization"
  ```

- [ ] **1.1.2** Install native environment
  ```bash
  cd /Users/tom/Documents/Git/eyeglass_finder
  poetry install
  poetry shell
  ```

- [ ] **1.1.3** Verify all dependencies work natively
  ```bash
  poetry run python -c "import torch, ultralytics, cv2; print('All imports successful')"
  poetry run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
  ```

- [ ] **1.1.4** Test basic pipeline functionality (CPU-only)
  ```bash
  # Force CPU mode for baseline
  export FORCE_CPU_ONLY=true
  poetry run python scripts/process_data.py --config config/config.yaml
  ```

- [ ] **1.1.5** Record CPU baseline performance
  ```
  Baseline Metrics to Record:
  - Processing time: _____ minutes
  - Throughput: _____ images/second  
  - Peak memory usage: _____ GB
  - CPU utilization: _____ %
  - Worker count: 8
  ```

- [ ] **1.1.6** Validate data integrity (compare with Docker results)
  - [ ] Check output file counts match
  - [ ] Verify no processing errors
  - [ ] Confirm all input images processed

### Task 1.2: Basic Algorithmic Optimizations (CPU-only) (2-4 hours)
**Objective:** Implement CPU-level optimizations before GPU acceleration

#### Subtasks:
- [ ] **1.2.1** Create performance benchmark utility
  ```python
  # Create: src/utils/benchmark.py
  class PerformanceBenchmark:
      def __init__(self):
          self.baseline_throughput = [RECORD_FROM_1.1.5]
          self.regression_threshold = 0.1  # 10% degradation threshold
      
      def validate_optimization(self, optimization_name):
          # Implementation as per optimization_notes.md
  ```

- [ ] **1.2.2** Optimize list building operations
  **Target:** `src/processing/pipeline.py` - Replace append loops with comprehensions
  - [ ] Identify all `for item in X: list.append()` patterns
  - [ ] Replace with list comprehensions: `[item for item in X if condition]`
  - [ ] Test with micro-benchmark (expect 15-25% improvement)

- [ ] **1.2.3** Optimize index calculations
  **Target:** Reduce repeated arithmetic in batch processing loops
  - [ ] Pre-compute index mappings: `np.arange(len(data)).reshape(batches, batch_size)`
  - [ ] Replace nested index calculations with lookup arrays
  - [ ] Test with micro-benchmark (expect 5-10% improvement)

- [ ] **1.2.4** Optimize memory allocations
  - [ ] Pre-size lists where possible: `[None] * expected_size`
  - [ ] Use NumPy arrays for numeric operations instead of Python lists
  - [ ] Add explicit garbage collection after large operations: `gc.collect()`

- [ ] **1.2.5** Validate algorithmic optimizations
  ```bash
  # Run with optimizations
  poetry run python scripts/process_data.py --config config/config.yaml
  ```
  ```
  Optimized CPU Metrics:
  - Processing time: _____ minutes (vs baseline: _____)
  - Throughput: _____ images/second (vs baseline: _____)
  - Improvement: _____ % 
  ```

### Task 1.3: Enhanced Configuration (1-2 hours)
**Objective:** Add M2 Max-specific configuration parameters

#### Subtasks:
- [ ] **1.3.1** Backup current config
  ```bash
  cp config/config.yaml config/config.yaml.backup
  ```

- [ ] **1.3.2** Add hardware-specific settings to `config/config.yaml`
  ```yaml
  # Add new section:
  hardware:
    max_workers: 10              # Phase 1 target (safer intermediate step)
    worker_memory_limit_mb: 1500 # Per-worker memory cap for 10 workers
    use_mps_acceleration: false  # Will enable in Task 1.4
    device_override: null        # Manual device override if needed
  ```

- [ ] **1.3.3** Add performance tuning parameters
  ```yaml
  # Add new section:
  performance:
    prefetch_chunks: 2           # Prefetch data for smoother streaming
    gc_frequency: 10             # Garbage collection every N chunks  
    thread_pool_workers: 4       # I/O thread pool size
    batch_optimization: true     # Enable new algorithmic optimizations
  ```

- [ ] **1.3.4** Add observability enhancements
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

- [ ] **1.3.5** Update code to use new configuration parameters
  **Files to modify:**
  - [ ] `src/processing/worker.py` - Add hardware.max_workers support
  - [ ] `scripts/process_data.py` - Add performance.gc_frequency support  
  - [ ] `src/utils/config.py` - Add validation for new parameters

- [ ] **1.3.6** Test configuration changes
  ```bash
  poetry run python scripts/process_data.py --config config/config.yaml --dry-run
  ```

### Task 1.4: MPS Integration + Comprehensive Validation (2-3 hours)
**Objective:** Enable Apple GPU acceleration with robust validation

#### Subtasks:
- [ ] **1.4.1** Create MPS validation utility
  ```python
  # Create: src/utils/mps_validator.py
  # Implement validation functions from optimization_notes.md:
  # - validate_yolov8_mps()
  # - validate_mps_memory() 
  # - validate_end_to_end_mps()
  ```

- [ ] **1.4.2** Run individual model validation
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

- [ ] **1.4.3** Run glasses-detector validation
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

- [ ] **1.4.4** Run memory usage validation
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

- [ ] **1.4.5** Enable MPS if validation passes
  **Condition:** All validations return compatible=True AND speed improvement >10%
  ```yaml
  # Update config/config.yaml:
  hardware:
    use_mps_acceleration: true  # Enable MPS
  ```

- [ ] **1.4.6** Run end-to-end MPS performance test
  ```bash
  poetry run python scripts/process_data.py --config config/config.yaml
  ```
  ```
  MPS Performance Results:
  - Processing time: _____ minutes
  - Throughput: _____ images/second
  - Improvement vs CPU baseline: _____ x
  - Memory usage: _____ GB
  - CPU utilization: _____ %
  ```

- [ ] **1.4.7** Implement automatic fallback (if MPS issues detected)
  **Target:** `src/utils/device.py`
  - [ ] Add performance monitoring during device selection
  - [ ] Implement fallback to CPU if MPS performance degrades >10%
  - [ ] Add logging for MPS fallback events

### Task 1.5: Phase 1 Validation & Documentation (30 minutes)
**Objective:** Validate Phase 1 success and document results

#### Subtasks:
- [ ] **1.5.1** Run performance regression check
  ```bash
  poetry run python -c "
  from src.utils.benchmark import PerformanceBenchmark
  benchmark = PerformanceBenchmark()
  benchmark.validate_optimization('Phase1-Complete')
  "
  ```

- [ ] **1.5.2** Document Phase 1 results
  ```
  PHASE 1 COMPLETION METRICS:
  
  Baseline (Docker):     _____ images/second
  CPU Optimizations:     _____ images/second (_____ % improvement)
  MPS Enabled:          _____ images/second (_____ % improvement)
  Total Phase 1 Gain:   _____ x (_____ % improvement)
  
  Memory Usage:         _____ GB (within _____ GB limit)
  Worker Count:         8 → 10 (if implemented)
  MPS Status:           _____ (Enabled/Disabled/Fallback)
  
  Success Criteria:
  - [ ] >50% improvement over baseline (target: 2-3x)
  - [ ] <50% memory increase
  - [ ] Zero processing errors
  - [ ] All validation tests pass
  ```

- [ ] **1.5.3** Create git checkpoint for Phase 1 completion
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

- [ ] **1.5.4** Prepare for Phase 2
  - [ ] Review Phase 2 requirements in `02_optimization_dev_guide.md`
  - [ ] Validate Phase 1 meets prerequisites for Phase 2
  - [ ] Document any issues or deviations for Phase 2 planning

## Success Criteria for Phase 1
- [ ] **Performance:** 2-3x improvement over baseline (minimum 50% improvement)
- [ ] **Stability:** Zero processing errors, all test cases pass
- [ ] **Memory:** <50% memory increase from baseline
- [ ] **MPS:** GPU acceleration working OR graceful fallback to CPU
- [ ] **Foundation:** Native execution environment fully operational

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
