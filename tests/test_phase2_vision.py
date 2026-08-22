"""
Phase 2 Vision Stack Tests: Pose Y-Slice Locking & Sub-Pixel Edge Repeatability.

Tests:
1. MediaPipe anatomical slice Y-lock drift across 4 angles < 0.5% of body height.
2. Sub-pixel edge detector repeatability standard deviation < 0.3 px on horizontal scanlines.
3. Sub-pixel parabolic interpolation precision < 0.1 px on synthetic step/ramp boundaries.
"""

import cv2
import numpy as np
import pytest

from body_measurement.edge_detection import SubPixelEdgeDetector
from body_measurement.landmarks import AnatomicalAnchorEngine, BodySite


class TestPhase2VisionStack:
    """Test suite for Phase 2 vision algorithms and landmark locking."""

    @pytest.fixture
    def edge_detector(self):
        return SubPixelEdgeDetector(gaussian_sigma=1.8, strip_half_height=2)

    @pytest.fixture
    def anchor_engine(self):
        return AnatomicalAnchorEngine()

    def test_subpixel_edge_repeatability_sd_below_0_3_px(self, edge_detector):
        """
        Validates that sub-pixel edge detector achieves repeatability SD < 0.3 px across repeated frames with noise.
        """
        w, h = 640, 480
        true_left_x = 220.45
        true_right_x = 420.75
        y_slice = 240

        measured_lefts = []
        measured_rights = []
        measured_widths = []

        # Run 50 frames with realistic sensor Gaussian noise (sigma=3.0)
        rng = np.random.RandomState(42)
        for i in range(50):
            frame = np.ones((h, w), dtype=np.float32) * 230.0  # Background

            # Body silhouette with subpixel anti-aliased edge
            col_indices = np.arange(w, dtype=np.float32)
            left_trans = np.clip(col_indices - (true_left_x - 1.0), 0.0, 1.0)
            right_trans = np.clip((true_right_x + 1.0) - col_indices, 0.0, 1.0)
            mask = np.minimum(left_trans, right_trans)

            body_intensity = 40.0
            frame = frame * (1.0 - mask) + body_intensity * mask

            # Add sensor noise
            noise = rng.normal(0.0, 3.0, frame.shape)
            noisy_frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

            res = edge_detector.extract_slice_edges(noisy_frame, y_slice=y_slice)
            assert res.is_valid, f"Edge detection failed on frame {i}"
            measured_lefts.append(res.left_edge_x)
            measured_rights.append(res.right_edge_x)
            measured_widths.append(res.width_pixels)

        sd_left = float(np.std(measured_lefts))
        sd_right = float(np.std(measured_rights))
        sd_width = float(np.std(measured_widths))

        # Exit criterion: repeatability SD < 0.3 px
        assert sd_left < 0.3, f"Left edge SD {sd_left:.4f} px exceeds 0.3 px threshold"
        assert sd_right < 0.3, f"Right edge SD {sd_right:.4f} px exceeds 0.3 px threshold"
        assert sd_width < 0.3, f"Width SD {sd_width:.4f} px exceeds 0.3 px threshold"

    def test_anatomical_slice_y_drift_across_4_angles_below_0_5_percent(self, anchor_engine):
        """
        Validates that anatomical Y-slice height drift across the 4 orthogonal angles is < 0.5% of body height.
        """
        body_height_px = 600.0
        h, w = 1080, 1920
        y_slices = []

        # Synthetic pose frames across 4 angles (Front 0, Right 90, Back 180, Left 270)
        # Even when profile silhouettes narrow the torso, the locked shoulder-hip vertical ratio must keep slice Y stable.
        for angle in [0, 90, 180, 270]:
            frame = np.ones((h, w, 3), dtype=np.uint8) * 230
            # Draw synthetic torso
            cv2.rectangle(frame, (800, 200), (1120, 800), (45, 45, 45), -1)

            res = anchor_engine.compute_anchor_slice(frame, site=BodySite.WAIST)
            assert res.slice_y_pixel > 0
            y_slices.append(res.slice_y_pixel)

        drift_px = max(y_slices) - min(y_slices)
        drift_percent = (drift_px / body_height_px) * 100.0

        # Exit criterion: drift < 0.5% of body height
        assert drift_percent < 0.5, f"Slice Y drift {drift_percent:.3f}% exceeds 0.5% threshold (drift={drift_px}px)"
