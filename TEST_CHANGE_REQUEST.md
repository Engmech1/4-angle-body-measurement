# Test Change Requests (TEST_CHANGE_REQUEST.md)

### Request TCR-001: Correct Invariant Direction for Convex Hull vs. Raw Anatomical Contour in Tier 1

- **Status**: PARKED (Awaiting Human Approval)
- **Target File**: `eval/tiers.py` (Tier 1 Analytic Math) & ANTIGRAVITY Build Spec §4 Tier 1
- **Claim**:
  The spec assertion in §4 Tier 1 stating `"hull perimeter >= raw perimeter always"` is mathematically inverted for non-convex curves.
- **Mathematical Justification**:
  For any closed, rectifiable 2D Jordan curve $\gamma$ in $\mathbb{R}^2$ with arc length $L(\gamma) = \text{perimeter\_raw}$, and its convex hull $\text{conv}(\gamma)$ with boundary perimeter $L(\partial \text{conv}(\gamma)) = \text{perimeter\_hull}$:
  $$\text{perimeter\_hull} \le \text{perimeter\_raw}$$
  with equality if and only if $\gamma$ is convex.
  A taut physical tape measure bridging a concave indentation (such as the posterior lumbar lordosis furrow or inter-gluteal cleft) follows straight line segments spanning the concavity, which by the triangle inequality is strictly shorter than the arc length traversing into the furrow:
  $$L_{\text{straight\_segment}} < L_{\text{concave\_path}}$$
- **Exposing Fixture**:
  Tier 1 Concave Lumbar Profile fixture:
  `twin_gen.generate_ground_truth_cross_section("lumbar_waist", 30.0, 20.0, superellipse_p=2.45, lordosis_depth_cm=3.0)`
  - Raw Contour Perimeter: $81.798\text{ cm}$
  - Convex Hull (Tape) Perimeter: $81.632\text{ cm}$
  - Delta: $\text{Raw} - \text{Hull} = +0.165\text{ cm} > 0$
- **Recommendation**:
  Update the test assertion in `run_tier1_analytic` and the spec text in §4 Tier 1 to assert:
  $$\text{perimeter\_hull} \le \text{perimeter\_raw} \quad (\text{with strict } \text{raw} > \text{hull} \text{ on concave fixtures})$$
- **Blocking**:
  Blocks modifying the formal Tier 1 test assertion until user review. (Item marked PARKED).
