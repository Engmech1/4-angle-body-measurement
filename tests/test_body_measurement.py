"""
Unit & Integration Test Suite for Body Measurement System.
Tests sub-pixel edge localization, ArUco metric scaling, landmark anchoring,
burst MAD filtering, spline perimeter reconstruction, and ML continual learning.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from body_measurement.adversarial_simulator import AdversarialSimulationConfig, AdversarialSimulator
from body_measurement.burst_processor import BurstAngleResult, BurstFrameProcessor
from body_measurement.edge_detection import SubPixelEdgeDetector
from body_measurement.landmarks import AnatomicalAnchorEngine, BodySite
from body_measurement.ml_optimizer import (
    AdaptiveMLReconstructor,
    BiomechanicalFeatureVector,
)
from body_measurement.reconstruction import (
    CrossSectionReconstructor,
    ReconstructionMethod,
)
from body_measurement.scaling import ArucoMetricScaler
from body_measurement.system import BodyMeasurementSystem, CaptureAngle


class TestArucoMetricScaler:
    def test_aruco_detection_and_subpixel(self):
        scaler = ArucoMetricScaler(marker_size_cm=10.0, dictionary_id=cv2.aruco.DICT_4X4_50)
        
        # Render a clean synthetic 10.0 cm ArUco marker
        marker_px = 200
        marker_img = np.zeros((marker_px, marker_px), dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker_img = cv2.aruco.generateImageMarker(dictionary, 0, marker_px, 1)
        else:
            marker_img = cv2.aruco.drawMarker(dictionary, 0, marker_px, 1)

        # Place marker on a larger white canvas
        canvas = np.full((600, 600), 255, dtype=np.uint8)
        canvas[200:400, 200:400] = marker_img

        res = scaler.detect_and_calibrate(canvas)
        assert res.is_valid is True
        assert res.marker_id == 0
        assert res.corners.shape == (4, 2)
        # 200 px / 10.0 cm = 20.0 pixels/cm
        assert abs(res.pixels_per_cm - 20.0) < 0.5
        assert res.scale_confidence > 0.8

    def test_depth_correction_factor(self):
        scaler = ArucoMetricScaler(marker_size_cm=10.0, fallback_pixels_per_cm=10.0)
        marker_img = np.full((600, 600), 255, dtype=np.uint8)
        
        # Test ratio calculation with fallback
        res = scaler.detect_and_calibrate(
            marker_img,
            distance_camera_to_wall_cm=300.0,
            distance_camera_to_subject_cm=200.0,
        )
        assert res.depth_correction_factor == 1.5


class TestSubPixelEdgeDetector:
    def test_exact_subpixel_boundary_recovery(self):
        detector = SubPixelEdgeDetector(gaussian_sigma=1.8, strip_half_height=2)
        
        w, h = 800, 200
        img = np.full((h, w), 200, dtype=np.uint8)

        # Exact sub-pixel boundaries
        exact_left = 215.35
        exact_right = 584.65
        exact_width = exact_right - exact_left  # 369.30 px

        # Render continuous smooth sigmoid transitions
        x_coords = np.arange(w, dtype=np.float64)
        sigma_edge = 2.0
        left_trans = 1.0 / (1.0 + np.exp(-(x_coords - exact_left) / sigma_edge))
        right_trans = 1.0 / (1.0 + np.exp(-(exact_right - x_coords) / sigma_edge))
        body = left_trans * right_trans
        profile = 200.0 - (150.0 * body)

        img[:, :] = np.clip(profile, 0, 255).astype(np.uint8)

        # Extract edges
        res = detector.extract_slice_edges(img, y_slice=100)
        assert bool(res.is_valid) is True
        # Check sub-pixel accuracy (< 0.25 pixel error)
        assert abs(res.left_edge_x - exact_left) < 0.25
        assert abs(res.right_edge_x - exact_right) < 0.25
        assert abs(res.width_pixels - exact_width) < 0.35


class TestBurstFrameProcessor:
    def test_sway_compensation_and_mad_filtering(self):
        processor = BurstFrameProcessor(mad_threshold=2.5)
        ppm = 10.0  # 10 px/cm
        
        # Simulate 30 frames with 2 cm center-of-mass sway + 2 outlier frames
        frames = []
        nominal_width_px = 300.0  # 30.0 cm
        
        for i in range(30):
            # Sway shifts center by up to 20 px (2.0 cm)
            sway_px = 15.0 * np.sin(2.0 * np.pi * i / 30.0)
            center = 400.0 + sway_px
            
            # 2 outlier frames with false shadow edge
            if i in [5, 18]:
                w_px = nominal_width_px + 50.0  # +5 cm error
            else:
                w_px = nominal_width_px + np.random.normal(0, 1.0)
                
            left_x = center - w_px / 2.0
            right_x = center + w_px / 2.0
            
            frame = np.full((100, 800), 220, dtype=np.uint8)
            frame[:, int(left_x):int(right_x)] = 50
            frames.append(frame)
            
        res = processor.process_burst(frames, y_slice=50, angle_degrees=0, pixels_per_cm=ppm)
        
        assert bool(res.is_valid) is True
        # Check that estimated width is within 0.25 cm of nominal 30.0 cm
        assert abs(res.width_cm - 30.0) < 0.25
        # Check that sway was tracked (~1.5 to 3.0 cm)
        assert res.center_sway_cm > 1.5


class TestCrossSectionReconstructor:
    def test_spline_vs_ground_truth_accuracy(self):
        simulator = AdversarialSimulator()
        gt = simulator.generate_ground_truth_anatomy(
            nominal_width_cm=32.0,
            nominal_depth_cm=22.0,
            lordosis_depth_cm=2.75,
            superellipse_p=2.45,
        )

        reconstructor = CrossSectionReconstructor(
            default_method=ReconstructionMethod.ANTHROPOMETRIC_LORDOSIS_SPLINE
        )

        res = reconstructor.reconstruct_cross_section(
            width_front_cm=gt.width_front_cm,
            depth_right_cm=gt.depth_right_cm,
            width_back_cm=gt.width_back_cm,
            depth_left_cm=gt.depth_left_cm,
            custom_lordosis_depth_cm=gt.lordosis_depth_cm,
            custom_superellipse_p=gt.superellipse_p,
        )

        error_cm = abs(res.perimeter_cm - gt.exact_perimeter_cm)
        assert error_cm < 0.10, f"Expected error < 0.10 cm, got {error_cm:.4f} cm"


class TestAdaptiveMLReconstructor:
    def test_feature_extraction_and_prediction(self):
        ml_opt = AdaptiveMLReconstructor()
        
        burst_data = {
            0: BurstAngleResult(0, 30, 30, 400.0, 1.2, 32.0, 1.5, 0.98, True),
            90: BurstAngleResult(90, 30, 30, 275.0, 1.1, 22.0, 1.8, 0.96, True),
            180: BurstAngleResult(180, 30, 30, 400.0, 1.0, 32.0, 1.4, 0.97, True),
            270: BurstAngleResult(270, 30, 30, 275.0, 1.3, 22.0, 1.6, 0.95, True),
        }
        
        features = ml_opt.extract_features(burst_data, torso_height_px=750.0, pixels_per_cm=12.5)
        assert features.coronal_width_front_cm == 32.0
        assert features.sagittal_depth_right_cm == 22.0
        assert abs(features.aspect_ratio - (32.0 / 22.0)) < 0.01
        
        # Test ML prediction
        res = ml_opt.predict_and_optimize(features, site=BodySite.WAIST)
        assert res.ml_corrected_perimeter_cm > 50.0
        assert res.estimated_uncertainty_cm > 0.0
        assert 2.25 <= res.adaptive_superellipse_p <= 2.65

    def test_online_continual_learning_update(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_model_path = tf.name

        try:
            ml_opt = AdaptiveMLReconstructor(model_path=temp_model_path)
            
            features = BiomechanicalFeatureVector(
                coronal_width_front_cm=32.0,
                coronal_width_back_cm=32.0,
                sagittal_depth_right_cm=22.0,
                sagittal_depth_left_cm=22.0,
                aspect_ratio=1.45,
                width_asymmetry_cm=0.0,
                depth_asymmetry_cm=0.0,
                mean_sway_amplitude_cm=1.5,
                max_sway_amplitude_cm=2.0,
                mean_edge_confidence=0.96,
                min_edge_confidence=0.92,
                torso_height_proxy_cm=60.0,
                base_perimeter_estimate_cm=87.0,
            )
            
            # True tape measurement check: 88.00 cm
            gt_tape = 88.00
            err_before = abs(ml_opt.predict_and_optimize(features).ml_corrected_perimeter_cm - gt_tape)
            
            # Apply online learning update
            ml_opt.online_update(features, ground_truth_perimeter_cm=gt_tape, learning_rate=0.20)
            
            err_after = abs(ml_opt.predict_and_optimize(features).ml_corrected_perimeter_cm - gt_tape)
            assert err_after < err_before, "Online learning should reduce error toward ground truth."
            assert os.path.exists(temp_model_path)
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)


class TestFullAdversarialQA:
    def test_pipeline_under_adversarial_noise(self):
        config = AdversarialSimulationConfig(
            pixels_per_cm=12.5,
            frames_per_angle=30,
            sway_amplitude_cm=1.5,
            edge_noise_pixels=4.5,
            occlusion_frame_prob=0.05,
        )
        simulator = AdversarialSimulator(config)
        
        gt = simulator.generate_ground_truth_anatomy(
            nominal_width_cm=30.0,
            nominal_depth_cm=20.0,
            lordosis_depth_cm=2.5,
            superellipse_p=2.45,
        )
        
        eval_res = simulator.evaluate_pipeline(gt)
        assert eval_res.passed_target_0_5cm is True, (
            f"Adversarial QA failed: error {eval_res.absolute_error_cm:.3f} cm exceeds 0.5 cm target."
        )
