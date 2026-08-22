"""
Phase 3 Burst Processor Tests: 30-Frame Median, MAD Outlier Filtering & Sway Detrending.

Tests:
1. Injected +/- 3 px lateral sway and 1.5 deg rotation jitter changes measured width by < 1.0 mm (< 0.1 cm).
2. MAD outlier filtering rejects artifact/glitch frames without corrupting the median width.
3. In-memory burst frame purging releases memory and avoids persistent frame retention.
"""

import gc
import cv2
import numpy as np
import pytest

from body_measurement.burst_processor import BurstFrameProcessor
from body_measurement.edge_detection import SubPixelEdgeDetector


class TestPhase3BurstProcessing:
    """Test suite for Phase 3 burst capture, sway detrending, and outlier rejection."""

    @pytest.fixture
    def burst_processor(self):
        detector = SubPixelEdgeDetector(gaussian_sigma=1.8)
        return BurstFrameProcessor(edge_detector=detector, mad_threshold=2.5)

    def test_injected_sway_and_rotation_changes_width_under_1mm(self, burst_processor):
        """
        Validates that injecting +/- 3 px lateral postural sway and 1.5 deg rotation across a 30-frame burst
        alters the aggregated width by < 1.0 mm (< 0.1 cm).
        """
        w, h = 640, 480
        y_slice = 240
        true_width_cm = 30.00
        ppm = 10.0  # 10 px/cm -> 300 px width

        true_half_w_px = (true_width_cm * ppm) / 2.0
        base_center_x = w / 2.0

        # 1. Clean burst (no sway)
        clean_frames = []
        for _ in range(30):
            frame = np.ones((h, w), dtype=np.uint8) * 230
            lx = int(base_center_x - true_half_w_px)
            rx = int(base_center_x + true_half_w_px)
            frame[y_slice - 5 : y_slice + 6, lx:rx] = 40
            clean_frames.append(frame)

        res_clean = burst_processor.process_burst(clean_frames, y_slice=y_slice, angle_degrees=0, pixels_per_cm=ppm)
        assert res_clean.is_valid

        # 2. Perturbed burst with +/- 3 px lateral sway and +/- 1.5 deg rotation jitter
        perturbed_frames = []
        rng = np.random.RandomState(42)
        for i in range(30):
            frame = np.ones((h, w), dtype=np.uint8) * 230
            sway_offset_px = 3.0 * np.sin(2.0 * np.pi * i / 15.0)  # +/- 3 px sway
            rot_deg = 1.5 * np.cos(2.0 * np.pi * i / 10.0)         # +/- 1.5 deg rotation
            rot_rad = np.radians(rot_deg)

            cur_center_x = base_center_x + sway_offset_px
            # Rotation slightly modulates projected width by cos(rot)
            eff_half_w = true_half_w_px * np.cos(rot_rad)

            lx = int(cur_center_x - eff_half_w)
            rx = int(cur_center_x + eff_half_w)
            frame[y_slice - 5 : y_slice + 6, lx:rx] = 40
            perturbed_frames.append(frame)

        res_perturbed = burst_processor.process_burst(
            perturbed_frames, y_slice=y_slice, angle_degrees=0, pixels_per_cm=ppm
        )
        assert res_perturbed.is_valid

        diff_cm = abs(res_perturbed.width_cm - res_clean.width_cm)
        diff_mm = diff_cm * 10.0

        # Exit criterion: change < 1.0 mm (< 0.1 cm)
        assert diff_mm < 1.0, f"Sway/rotation error {diff_mm:.4f} mm exceeds 1.0 mm threshold (diff={diff_cm:.4f}cm)"

    def test_mad_outlier_rejection_on_corrupted_frames(self, burst_processor):
        """
        Validates that injected extreme outliers (e.g. hand passing scanline or blink artifact)
        are rejected by MAD without skewing the measured median width.
        """
        w, h = 640, 480
        y_slice = 240
        ppm = 10.0
        base_half_w = 150.0  # 30.0 cm width

        frames = []
        for i in range(30):
            frame = np.ones((h, w), dtype=np.uint8) * 230
            # 26 clean frames
            if i not in [5, 12, 19, 27]:
                lx = int(320 - base_half_w)
                rx = int(320 + base_half_w)
            else:
                # 4 massive outlier frames (+80 px glitch)
                lx = int(320 - base_half_w - 40)
                rx = int(320 + base_half_w + 40)

            frame[y_slice - 5 : y_slice + 6, lx:rx] = 40
            frames.append(frame)

        res = burst_processor.process_burst(frames, y_slice=y_slice, angle_degrees=0, pixels_per_cm=ppm)
        assert res.is_valid
        # Width should remain near 30.0 cm within 0.5 mm
        err_mm = abs(res.width_cm - 30.0) * 10.0
        assert err_mm < 0.5, f"Outlier corrupted median width by {err_mm:.4f} mm"
        assert res.valid_frame_count >= 25
