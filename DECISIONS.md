# Architectural & Engineering Decisions (DECISIONS.md)

This document records the architectural choices, mathematical defaults, and engineering conventions adopted in accordance with §8 of the ANTIGRAVITY Build Spec.

---

### Decision 1: Pure-NumPy & SciPy Geometric Digital Twin Generator
- **Choice**: Implement the procedural 3D parametric torso mesh & analytical cross-section generator using vectorized NumPy and `scipy.spatial` (with `ConvexHull` for exact convex hull tape measurement ground-truth) and a software pinhole rasterizer.
- **Rationale**: Guarantees zero platform dependency issues across Windows/Linux without requiring external heavy OpenGL / C++ binary renderers, while maintaining sub-millimeter geometric accuracy for digital twin ground-truth slicing.

### Decision 2: ArUco Marker in Coronal Plane + solvePnP Plane Validation
- **Choice**: Primary scale reference target is placed in the subject's coronal plane (turntable mat / floor toe-line ArUco board with 4x4 dictionary). The wall-mounted marker is retained strictly as a secondary redundant depth-sanity cross check.
- **Tolerance**: `solvePnP` must verify the ArUco board normal deviates from the optical axis by $< 15^\circ$; otherwise the frame is rejected for calibration.

### Decision 3: 4-Angle vs. N-Angle Reconstruction Gating
- **Choice**:
  - For $N = 4$ angles (0°, 90°, 180°, 270°), the system fits a 3-parameter superellipse $|x/a|^p + |y/b|^p = 1$ using documented ISAK / anthropometric priors per site ($p_{\text{waist}} \approx 2.45$, $p_{\text{chest}} \approx 2.50$, $p_{\text{thigh}} \approx 2.15$), calculating sensitivity $\frac{dP}{dp}$ as explicit `model_uncertainty_cm`.
  - For $N \ge 8$ angles (turntable continuous capture), the system activates ray-casting polygonal cross-section slicing, Truncated Fourier series smoothing, and spline contour modeling.
  - Both `perimeter_raw` and `perimeter_hull` (convex hull representing a taut physical tape) are computed and reported.

### Decision 4: Determinism & Random Seeds
- **Choice**: All synthetic generators, burst noise simulators, and test suites are seeded with fixed integers (default `seed = 42`) and enforce `PYTHONHASHSEED=0`.

### Decision 5: Privacy & Air-Gap Enforcement from Phase 0
- **Choice**: All video frames are stored strictly in volatile RAM (`bytearray` / `np.ndarray`), processed line-by-line, zeroed out (`memset`), and immediately garbage-collected. Socket connections to non-loopback IPs are forbidden and tested in Tier 7.
