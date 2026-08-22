# Physics & Mathematical Error Budget (ERROR_BUDGET.md)

**System Accuracy Target:** Total Combined Circumference Error $\le 5.0\text{ mm}$ ($\pm 0.5\text{ cm}$) compared to ground-truth taut tape measure perimeter (Convex Hull).

---

## 1. Pixel Budget & Optical Geometry

Let the camera sensor have vertical resolution $H$ pixels framing a subject field of view $H_{\text{FOV}} = 2.0\text{ m}$ (2000 mm) at distance $Z \approx 2.2\text{ m}$.

$$\text{Pixel Scale} = \frac{H_{\text{FOV}}}{H}$$

- **1080p Mode ($1920 \times 1080$):**
  $$\text{Scale}_{1080p} \approx \frac{2000\text{ mm}}{1920\text{ px}} \approx 1.04\text{ mm/pixel}$$
- **4K Mode ($3840 \times 2160$):**
  $$\text{Scale}_{4K} \approx \frac{2000\text{ mm}}{3840\text{ px}} \approx 0.52\text{ mm/pixel}$$

---

## 2. Error Component Breakdown (mm)

| Error Source | Physics / Mechanism | 1080p (4-Angle) | 4K (8+ Angle) | Mitigation in Pipeline |
|---|---|---|---|---|
| **$\sigma_{\text{calib}}$ (Scale Calibration)** | ArUco sub-pixel corner jitter, distance ratio $Z_{\text{wall}}/Z_{\text{subject}}$ uncertainty | $0.80\text{ mm}$ | $0.40\text{ mm}$ | Subject-plane ArUco board + `solvePnP` tilt check ($\le 15^\circ$) |
| **$\sigma_{\text{dist}}$ (Lens Distortion)** | Radial/tangential optical distortion residual | $0.50\text{ mm}$ | $0.30\text{ mm}$ | `cv2.undistort` with calibrated ChArUco polynomial matrix |
| **$\sigma_{\text{seg}}$ (Segmentation Bias)** | Alpha boundary transition & edge contrast gradient | $1.20\text{ mm}$ | $0.70\text{ mm}$ | SAM / DoG continuous 1D zero-crossing locator |
| **$\sigma_{\text{edge}}$ (Sub-pixel Noise)** | Spatial discrete sampling noise | $0.30\text{ mm}$ | $0.15\text{ mm}$ | 30-frame burst median filter + MAD outlier rejection |
| **$\sigma_{\text{landmark}}$ (Y-Slice Drift)** | MediaPipe pose jitter along vertical body axis | $1.00\text{ mm}$ | $0.60\text{ mm}$ | Normalized torso height lock (inter-shoulder to hip anchor) |
| **$\sigma_{\text{model}}$ (Shape Modeling)** | Prior superellipse exponent $p$ deviation vs true shape | $1.80\text{ mm}$ | $0.50\text{ mm}$ | Site-specific priors (4-angle) / Fourier polygon (8+ angle) |
| **$\sigma_{\text{posture}}$ (Protocol / Respiration)** | End-tidal exhale variation, sway, stance stability | $2.00\text{ mm}$ | $2.00\text{ mm}$ | Real-time HUD pose gating & auditory breathing cues |

---

## 3. Total Error Quadrature Combination

$$\sigma_{\text{total}} = \sqrt{\sigma_{\text{calib}}^2 + \sigma_{\text{dist}}^2 + \sigma_{\text{seg}}^2 + \sigma_{\text{edge}}^2 + \sigma_{\text{landmark}}^2 + \sigma_{\text{model}}^2 + \sigma_{\text{posture}}^2}$$

### Calculation for 1080p (4-Angle Mode)
$$\sigma_{\text{total, 1080p}} = \sqrt{0.80^2 + 0.50^2 + 1.20^2 + 0.30^2 + 1.00^2 + 1.80^2 + 2.00^2}$$
$$\sigma_{\text{total, 1080p}} = \sqrt{0.64 + 0.25 + 1.44 + 0.09 + 1.00 + 3.24 + 4.00} = \sqrt{10.66} \approx \mathbf{3.26\text{ mm}} < \mathbf{5.00\text{ mm}}$$

### Calculation for 4K (8+ Angle Turntable Mode)
$$\sigma_{\text{total, 4K}} = \sqrt{0.40^2 + 0.30^2 + 0.70^2 + 0.15^2 + 0.60^2 + 0.50^2 + 2.00^2}$$
$$\sigma_{\text{total, 4K}} = \sqrt{0.16 + 0.09 + 0.49 + 0.0225 + 0.36 + 0.25 + 4.00} = \sqrt{5.3725} \approx \mathbf{2.32\text{ mm}} < \mathbf{5.00\text{ mm}}$$

---

## 4. Conclusion & Gate Feasibility
The physical error budget **closes** below the $5.0\text{ mm}$ target for both 1080p and 4K configurations under the condition that protocol guidelines (end-tidal exhale, pose gating) and subject-plane calibration are strictly enforced.
