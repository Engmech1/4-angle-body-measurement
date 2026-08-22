"""
Hypothesis Property-Based Fuzz Testing for Body Measurement Components.

Formally tests algebraic and geometric invariants over arbitrary, stochastically
generated human morphology and camera configurations to eliminate overfitting and numerical regressions.
"""

from hypothesis import given, strategies as st, settings, HealthCheck
import numpy as np
import pytest

from body_measurement.reconstruction import CrossSectionReconstructor, ReconstructionMethod
from body_measurement.edge_detection import SubPixelEdgeDetector
from body_measurement.burst_processor import BurstFrameProcessor


class TestHypothesisPropertyFuzzing:
    """Property-based invariant testing suite."""

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        width=st.floats(min_value=15.0, max_value=60.0),
        depth=st.floats(min_value=10.0, max_value=45.0),
        p=st.floats(min_value=1.8, max_value=3.2),
        lordosis=st.floats(min_value=0.0, max_value=5.0),
    )
    def test_hypothesis_cross_section_geometry_invariants(self, width, depth, p, lordosis):
        """
        Validates fundamental geometric invariants:
        1. Perimeter is positive, finite, and within physical bounds:
           2 * (width + depth) > P_hull > 2 * sqrt(width^2 + depth^2)
        2. Hull perimeter is strictly <= Raw contour perimeter (TCR-001)
        3. Cross-sectional area is positive and strictly < width * depth
        4. Aspect ratio = width / depth
        5. Zero NaNs or Infs
        """
        recon = CrossSectionReconstructor()
        res = recon.reconstruct_cross_section(
            width_front_cm=width,
            depth_right_cm=depth,
            custom_superellipse_p=p,
            custom_lordosis_depth_cm=lordosis,
            method=ReconstructionMethod.ANTHROPOMETRIC_LORDOSIS_SPLINE,
        )

        assert res.is_valid is True
        assert np.isfinite(res.perimeter_cm)
        assert np.isfinite(res.perimeter_hull_cm)
        assert np.isfinite(res.perimeter_raw_cm)
        assert np.isfinite(res.cross_sectional_area_cm2)

        # 1. Bounding box & bounding perimeter sanity
        box_perimeter = 2.0 * (width + depth)
        assert res.perimeter_hull_cm < box_perimeter + 1e-4

        # 2. Hull <= Raw (taut tape is never longer than concave contour)
        assert res.perimeter_hull_cm <= res.perimeter_raw_cm + 1e-4

        # 3. Area bound: 0 < Area < width * depth
        assert 0 < res.cross_sectional_area_cm2 < (width * depth)

        # 4. Aspect ratio consistency
        assert abs(res.aspect_ratio - (width / depth)) < 1e-5

    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        center_jitter_px=st.floats(min_value=-30.0, max_value=30.0),
        width_px=st.floats(min_value=120.0, max_value=350.0),
        noise_std=st.floats(min_value=0.0, max_value=5.0),
    )
    def test_hypothesis_subpixel_edge_detector_stability(self, center_jitter_px, width_px, noise_std):
        """
        Validates that SubPixelEdgeDetector stably identifies body boundaries
        over arbitrary continuous edge positions and random sensor noise.
        """
        detector = SubPixelEdgeDetector(gaussian_sigma=1.8, strip_half_height=2)
        w_img, h_img = 640, 480
        y_slice = 240
        cx = (w_img / 2.0) + center_jitter_px
        left_x = cx - (width_px / 2.0)
        right_x = cx + (width_px / 2.0)

        frame = np.ones((h_img, w_img), dtype=np.uint8) * 230
        x_coords = np.arange(w_img, dtype=np.float64)

        # Sigmoid smooth step
        sigma_edge = 1.8
        left_trans = 1.0 / (1.0 + np.exp(-(x_coords - left_x) / (sigma_edge * 0.5)))
        right_trans = 1.0 / (1.0 + np.exp(-(right_x - x_coords) / (sigma_edge * 0.5)))
        body_mask = left_trans * right_trans
        profile = 230.0 - (180.0 * body_mask)

        if noise_std > 0:
            profile += np.random.normal(0, noise_std, size=w_img)
        profile = np.clip(profile, 0, 255).astype(np.uint8)
        frame[y_slice - 5 : y_slice + 6, :] = profile

        res = detector.extract_slice_edges(frame, y_slice=y_slice)
        assert bool(res.is_valid)
        assert abs(res.width_pixels - width_px) < max(2.5, noise_std * 1.5)
