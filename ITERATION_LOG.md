# ANTIGRAVITY Engineering Iteration Log (ITERATION_LOG.md)

This log tracks every iteration, git state, benchmark metric, root-cause diagnosis, and hypothesis verification in accordance with §6 of the ANTIGRAVITY Build Spec.

---

### Phase 0: Test Harness, Digital Twin & Error Budget Baseline

- **Iteration**: `00`
- **Objective**: Establish the 8-tier benchmark oracle, procedural 3D digital twin generator, camera calibration tool, and closed-form physical error budget table.
- **Scoreboard Summary**:
  - **Tier 1 (Analytic Math)**: MAE = 0.000 cm, P95 = 0.000 cm (PASS)
  - **Tier 2 (Digital Twin - DEV)**: MAE = 87.923 cm, Refusal = 100.0% (Deliberate baseline failure: pipeline stubbed)
  - **Tier 2 (Digital Twin - HOLDOUT)**: MAE = 87.923 cm, Refusal = 100.0% (Deliberate baseline failure: pipeline stubbed)
  - **Tier 3 (Metamorphic Invariance)**: 8/8 Passed (PASS)
  - **Tier 4 (Adversarial Robustness)**: 5/5 Passed (PASS)
  - **Tier 5 (Physical Proxies)**: MAE = 0.116 cm (PASS)
  - **Tier 6 (Human Test-Retest)**: NOT_RUN (No human rig active)
  - **Tier 7 (Privacy & Air-Gap)**: 3/3 Passed (PASS)
  - **Tier 8 (Golden File Canary)**: 1/1 Passed (PASS)
- **Artifact Generated**: `artifacts/metrics_iter_00.json`
- **Phase 0 Status**: **COMPLETED & VERIFIED**.

---

### Phase 1: Metric Scaling & 3D ArUco Calibration Gate

- **Iteration**: `01`
- **Objective**: Implement subpixel ArUco fiducial scaling, solvePnP 3D plane normal estimation, tilt angle rejection (>15 deg), and 20.00 cm reference bar validation across multiple distances and tilts.
- **Verification**:
  - `tests/test_scaling.py`: 3/3 Passed.
  - 20.00 cm reference bar reads within $< 0.1\text{ mm}$ error (well below $\pm 0.5\text{ mm}$ threshold) at $1.8\text{ m}, 2.2\text{ m}, 3.0\text{ m}$ and $0^\circ, 5^\circ, 10^\circ$ tilts.
  - solvePnP plane normal estimation successfully rejects tilt $> 15^\circ$.
- **Phase 1 Status**: **COMPLETED & VERIFIED**.

---

### Phase 2: Pose Landmarks, Y-Slice Lock & Sub-Pixel Edge Detection

- **Iteration**: `02`
- **Objective**: Verify MediaPipe 33-landmark pose anchoring, invariant Y-slice locking across 4 angles, and sub-pixel edge detection repeatability.
- **Verification**:
  - `tests/test_phase2_vision.py`: 2/2 Passed.
  - Sub-pixel edge detector repeatability standard deviation: $\sigma_{\text{edge}} < 0.15\text{ px}$ (well below $0.3\text{ px}$ exit criterion).
  - Anatomical slice Y-lock vertical drift across 4 angles: $0.0\% < 0.5\%$ body height.
- **Phase 2 Status**: **COMPLETED & VERIFIED**.

---

### Phase 3: Burst Capture, MAD Outlier Filtering & Sway Detrending

- **Iteration**: `03`
- **Objective**: Implement 30-frame burst processing, postural sway detrending, and MAD outlier rejection.
- **Verification**:
  - `tests/test_phase3_burst.py`: 2/2 Passed.
  - Injected $\pm 3\text{ px}$ lateral sway and $1.5^\circ$ rotational jitter altered aggregated width by $< 0.4\text{ mm}$ (well below $1.0\text{ mm}$ exit criterion).
  - MAD Modified Z-score filtering successfully rejected corrupted outlier frames with error $< 0.05\text{ mm}$.
- **Phase 3 Status**: **COMPLETED & VERIFIED**. Proceeding to Phase 4 (Cross-Section Reconstruction & Perimeter Optimization).



