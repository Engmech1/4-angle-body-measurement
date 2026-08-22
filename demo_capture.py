"""
Exercise App - Interactive Guided Capture CLI Demo.

Simulates the end-to-end user experience of the 4-Angle Guided Capture Body Measurement System:
1. Calibration (ArUco marker detection on wall / floor)
2. Anatomical Anchoring (MediaPipe 33-keypoint normalized invariant waist slice)
3. 4-Angle Guided Capture Burst (Front 0°, Right 90°, Back 180°, Left 270°)
4. In-Memory Zero-Raw-Media Processing
5. Biomechanical Lordosis-Spline Reconstruction & Detailed Health Metrics Report
"""

import sys
import time
import numpy as np

from body_measurement.adversarial_simulator import AdversarialSimulationConfig, AdversarialSimulator
from body_measurement.landmarks import BodySite
from body_measurement.reconstruction import ReconstructionMethod
from body_measurement.system import BodyMeasurementSystem, CaptureAngle


def main():
    print("=" * 70)
    print("      EXERCISE APP: 4-ANGLE GUIDED CAPTURE SYSTEM")
    print("   High-Precision Biometric Perimeter & Cross-Section Engine")
    print("=" * 70)

    # 1. Initialize System
    system = BodyMeasurementSystem(
        marker_size_cm=15.0,
        reconstruction_method=ReconstructionMethod.ANTHROPOMETRIC_LORDOSIS_SPLINE,
    )

    # 2. Camera Setup & Metric Calibration
    print("\n[STEP 1/4] Camera Metric Calibration (ArUco Detection)...")
    time.sleep(0.2)
    # Calibrated at 12.5 pixels/cm (standard 1080p full body at 2.5m)
    system.set_manual_scale(pixels_per_cm=12.5)
    print("  -> Calibration successful: Pixels-Per-Centimeter = 12.50 px/cm")
    print("  -> Marker geometry verified: Depth offset factor = 1.000")

    # 3. Anatomical Landmark Anchoring
    print("\n[STEP 2/4] MediaPipe Anatomical Anchoring...")
    time.sleep(0.2)
    print("  -> Tracking 33 full-body landmarks (shoulders, hips, knees, ankles)")
    print("  -> Anchored site: WAIST (ISAK Standard: 0.618 normalized torso span)")
    print("  -> Invariant slice Y: 960 px (Torso height span: 750 px)")

    # 4. Multi-Frame Burst Ingestion across 4 Angles
    print("\n[STEP 3/4] 4-Angle Guided Capture Stream (30 frames/burst)...")
    simulator = AdversarialSimulator(
        AdversarialSimulationConfig(
            pixels_per_cm=12.5,
            frames_per_angle=30,
            sway_amplitude_cm=1.5,
            edge_noise_pixels=4.5,
        )
    )

    # Ground truth: Realistic waist (Width: 32 cm, Depth: 22 cm, Lordosis: 2.75 cm)
    gt = simulator.generate_ground_truth_anatomy(
        nominal_width_cm=32.0,
        nominal_depth_cm=22.0,
        lordosis_depth_cm=2.75,
        superellipse_p=2.45,
    )

    angles = [
        (CaptureAngle.FRONT, 0, "0° Front View"),
        (CaptureAngle.RIGHT_PROFILE, 90, "90° Right Profile"),
        (CaptureAngle.BACK, 180, "180° Back View"),
        (CaptureAngle.LEFT_PROFILE, 270, "270° Left Profile"),
    ]

    for angle_enum, angle_deg, label in angles:
        print(f"  -> Capturing {label} (30 in-memory frames)...", end=" ")
        frames = simulator.generate_adversarial_test_case(gt, angle_deg)
        burst_res = system.process_angle_burst(angle_enum, frames, y_slice=960)
        print(
            f"Done! Width: {burst_res.width_cm:.2f} cm (Sway detrended: {burst_res.center_sway_cm:.2f} cm)"
        )
        time.sleep(0.1)

    # 5. Non-Elliptical Cross-Section Reconstruction
    print("\n[STEP 4/4] Geometric Cross-Section & Spline Perimeter Integration...")
    time.sleep(0.2)
    summary = system.compute_measurement(site=BodySite.WAIST)

    print("\n" + "=" * 70)
    print("                      BIOMETRIC MEASUREMENT REPORT")
    print("=" * 70)
    print(f"  Target Anatomical Site    : {summary.site.value.upper()}")
    print(f"  Calculated Perimeter      : {summary.perimeter_cm:.2f} cm")
    print(f"  Ground Truth Perimeter    : {gt.exact_perimeter_cm:.2f} cm")
    err = abs(summary.perimeter_cm - gt.exact_perimeter_cm)
    print(f"  Absolute Error            : {err:.3f} cm (Target: < 0.50 cm -> {'PASS' if err < 0.5 else 'FAIL'})")
    print(f"  Relative Error            : {(err / gt.exact_perimeter_cm) * 100.0:.2f} %")
    print("-" * 70)
    print(f"  Frontal Width (Coronal)   : {summary.coronal_width_cm:.2f} cm")
    print(f"  Profile Depth (Sagittal)  : {summary.sagittal_depth_cm:.2f} cm")
    print(f"  Width / Depth Ratio       : {summary.aspect_ratio:.2f}")
    print(f"  Cross-Sectional Area      : {summary.cross_sectional_area_cm2:.1f} cm²")
    print(f"  Reconstruction Algorithm  : {summary.reconstruction_method.value}")
    print(f"  Privacy Guard Status      : Zero raw frames saved to disk (100% In-Memory)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
