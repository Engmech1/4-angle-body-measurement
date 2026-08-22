# ANTIGRAVITY Iteration Log & Benchmark History

## Phase 0: Ground-Truth Harness & Architectural Firewall Verification
- **Status**: PASSED / EXITED
- **Timestamp**: 2026-08-23T02:03:00+07:00
- **Import Firewall Status (`tests/test_import_firewall.py`)**: PASSED (100% compliant, zero forbidden imports in `body_measurement/**`)
- **Package Versions Verified**:
  - `python`: 3.13.2
  - `numpy`: 2.5.0
  - `scipy`: 1.18.0
  - `opencv-python`: 5.0.0.93
  - `mediapipe`: 1.0.1
  - `hypothesis`: 6.165.10
  - `albumentations`: 2.0.8
  - `imagecorruptions`: 1.1.2
  - `trimesh`: 5.0.0
  - `anny`: 0.6.0
  - `clad-body`: 0.6.1
- **Phase 0 Artifacts Generated**:
  - `smoke_body_mesh.obj`: 5,120 vertices, 10,112 faces
  - `smoke_ground_truth.json`: Waist (94.998 cm), Chest (103.311 cm), Hips (100.037 cm)
  - `smoke_render_4view.png`: 1280x960 px 4-angle projection matrix
- **Benchmark Suite**: 8 Tiers initialized (T0-T7)

---

## Iteration History

### Iteration 00 (Baseline)
- **Date**: 2026-08-23
- **Commit/Tag**: `iter-00`
- **Hypothesis**: Baseline Superellipse (p=2.45) with 4-angle silhouette projection against Ground Truth 3D Convex Hull.
- **Scoreboard Summary**: Baseline Phase 0 initial harness. Tier 2 MAE 56.97 cm due to ArUco marker collision with waist scanline.

---

### Iteration 01 (Item 1: Holdout Rebuild & Parameter Sweep)
- **Date**: 2026-08-23
- **Commit**: `deecca6` / `iter-01`
- **Work-Queue Item**: `Item 1: Fix fake holdout split`
- **Hypothesis / Change**:
  Rebuilt `eval/synthetic_generator.py` to produce strictly disjoint DEV (5 subjects) and HOLDOUT (5 subjects) morphology splits.
  Implemented full §4 Tier 2.5 camera parameter sweep on holdout:
  - Camera height: 85–115 cm ($\pm 15$ cm)
  - Subject distance: 1.85–3.45 m
  - Focal length: 0.75x–1.5x ($\times 3$)
  - Camera roll: $-1.8^\circ$ to $+1.9^\circ$
  - Marker tilt: $3^\circ$ to $19^\circ$
  - Resolution: 1080p and 4K (3840x2160)
  Fixed ArUco collision by moving marker to top-left wall (`marker_x = 60, marker_y = 60`), preventing edge detector interference with the waist flank.
  Generated `artifacts/holdout_manifest.json` with SHA-256 hash `09cdb9251de01ff89df8c86e1b48bc484ccd85eb70f50305d8c3709b480c690d`.
  Added `tests/test_holdout_integrity.py` with 100% passing integrity and disjointness tests.
- **Artifact**: `artifacts/metrics_iter_01.json`
- **Scoreboard Result**:
  - Tier 1: PASS (MAE 0.000 cm)
  - Tier 2 (DEV): PASS (MAE 0.196 cm, Bias 0.196 cm, P95 0.435 cm)
  - Tier 2 (HOLDOUT): MAE 0.451 cm, Bias 0.446 cm
  - Tiers 3, 4, 5, 7, 8: PASS

---

### Iteration 02 (Item 2 & 3: Tier 2 Error Diagnosis & Quality Gating)
- **Date**: 2026-08-23
- **Commit**: `iter-02`
- **Work-Queue Items**: `Item 2: Tier 2 MAE/Bias instrumentation` & `Item 3: Quality Gating & Silent Failure Elimination`
- **Instrumentation & Diagnostic Findings**:
  - Damped intermediate values:
    - Subject dev_01: GT W=28.00 cm, Rec W=28.05 cm ($+0.5\text{ mm}$ error)
    - Subject dev_01: GT D=19.50 cm, Rec D=19.65 cm ($+1.5\text{ mm}$ error)
  - Previous 56.97 cm error was 100% caused by ArUco marker rendered at $y = \text{waist\_y}$ across the flank.
  - Implemented strict quality sanity gating in `body_measurement/system.py`:
    - Human girth bounds: $[40.0\text{ cm}, 250.0\text{ cm}]$
    - Coronal width bounds: $[15.0\text{ cm}, 90.0\text{ cm}]$
    - Sagittal depth bounds: $[10.0\text{ cm}, 70.0\text{ cm}]$
    - Aspect ratio bounds: $[0.60, 2.50]$
    - Front/Back symmetry tolerance: $\Delta W \le 8.0\text{ cm}$
    - Left/Right depth symmetry tolerance: $\Delta D \le 6.0\text{ cm}$
    - Corrupted or anomalous frames return `is_successful = False` with explicit `quality_flags`.
- **Artifact**: `artifacts/metrics_iter_02.json`
- **Scoreboard Result**:
  - Silent failure rate on DEV: 0.0% (PASS)
  - Refusal rate: 0.0%

---

### Iteration 03 (Item 4: Fix Ground Truth Raw Polyline Length in Smoke Test)
- **Date**: 2026-08-23
- **Commit**: `iter-03`
- **Work-Queue Item**: `Item 4: Fix perimeter_raw ~30x factor error in eval/smoke.py`
- **Root Cause & Fix**:
  - `mesh.section(...).vertices` in `trimesh` returns unordered line segment endpoints. Applying `np.diff` walked a zigzag path across cross-sections, giving 2974.6 cm (29.7 m).
  - Replaced with ordered planar polyline length: `slice_2d, _ = lines.to_planar(); perimeter_raw = float(slice_2d.length * 100.0)`.
  - Re-exported `artifacts/smoke_ground_truth.json`:
    - WAIST: TapeHull = 94.998 cm, RawContour = 94.999 cm
    - CHEST: TapeHull = 103.311 cm, RawContour = 103.311 cm
    - HIPS: TapeHull = 100.037 cm, RawContour = 100.037 cm
- **Artifact**: `artifacts/metrics_iter_03.json`

---

### Iteration 04 (Item 5 & Continuous Improvement: Adaptive Power & Girth Scaling)
- **Date**: 2026-08-23
- **Commit**: `iter-04`
- **Work-Queue Item**: `Item 5: Park hull >= raw spec contradiction (TCR-001)` & `Adaptive Morphological Power Tuning`
- **Changes**:
  - Created `TEST_CHANGE_REQUEST.md` (TCR-001) documenting the mathematical proof that $\text{Hull} \le \text{Raw}$ for concave contours and marking the item `PARKED`.
  - Added girth-dependent adaptive power estimation $\Delta p(r_{\text{mean}})$ in `CrossSectionReconstructor._estimate_adaptive_power`.
  - Set default reconstruction method to `DEFORMABLE_SUPERELLIPSE` for standard taut tape measurements.
- **Artifact**: `artifacts/metrics_iter_04.json`
