"""
Exercise App - Adversarial Simulation & Verification Harness.

Runs the complete 4-Angle Guided Capture Body Measurement Pipeline against
challenging adversarial synthetic inputs simulating real-world human sway (1-2 cm),
segmentation edge noise (+/- 5 px), soft shadows, and non-elliptical (kidney/bean)
lumbar-lordosis cross sections.

Verifies that the absolute error is strictly < 0.5 cm across all anthropometric body shapes.
"""

import sys
import time
from typing import List, Tuple
import numpy as np

from body_measurement.adversarial_simulator import (
    AdversarialSimulationConfig,
    AdversarialSimulator,
    GroundTruthCrossSection,
    SimulationEvaluationResult,
)
from body_measurement.reconstruction import ReconstructionMethod


def print_banner(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def run_single_adversarial_experiment(
    simulator: AdversarialSimulator,
    name: str,
    width_cm: float,
    depth_cm: float,
    lordosis_cm: float,
    superellipse_p: float,
) -> Tuple[GroundTruthCrossSection, SimulationEvaluationResult]:
    """Generates ground truth and evaluates the pipeline under severe noise."""
    gt = simulator.generate_ground_truth_anatomy(
        nominal_width_cm=width_cm,
        nominal_depth_cm=depth_cm,
        lordosis_depth_cm=lordosis_cm,
        superellipse_p=superellipse_p,
    )
    res = simulator.evaluate_pipeline(gt)
    return gt, res


def main():
    print_banner("4-Angle Guided Capture Body Measurement System: Adversarial QA Benchmark")
    print("Engine: OpenCV + MediaPipe + Sub-Pixel DoG + Anthropometric Lordosis Spline")
    print("Target Accuracy: Outperform human tape measure (ISAK TEM 1-2%). Target Error < 0.50 cm.")
    print("Privacy: 100% In-Memory Processing (Zero-Raw-Media).")

    # Configure adversarial environment:
    # 1.5 cm center-of-mass sway, +/- 4.5 px edge noise, 6.0 px soft shadow blur
    config = AdversarialSimulationConfig(
        pixels_per_cm=12.5,
        image_width=1080,
        image_height=1920,
        frames_per_angle=30,
        sway_amplitude_cm=1.5,
        edge_noise_pixels=4.5,
        shadow_softness_pixels=6.0,
        angle_jitter_deg=2.0,
    )

    simulator = AdversarialSimulator(config)

    # Test cases representing diverse human somatotypes
    test_cases = [
        # (Name, Width cm, Depth cm, Lumbar Lordosis cm, Superellipse Power p)
        ("Athletic / V-Taper Waist", 28.0, 19.5, 2.80, 2.40),
        ("Average Adult Waist", 32.0, 22.0, 2.75, 2.45),
        ("Heavy / Android Waist", 38.5, 30.0, 2.20, 2.55),
        ("Slender / Ectomorph Waist", 25.5, 17.0, 2.90, 2.35),
        ("Deep Lumbar Lordosis Profile", 30.0, 21.0, 3.50, 2.40),
        ("Flat Back Posture Profile", 31.0, 21.5, 1.40, 2.50),
    ]

    results: List[Tuple[str, GroundTruthCrossSection, SimulationEvaluationResult]] = []
    errors: List[float] = []

    print(f"\nRunning {len(test_cases)} Adversarial Somatotype Test Cases (30 frames/angle = 120 frames each)...")
    print("-" * 75)
    print(f"{'Somatotype':<30} | {'GT (cm)':<8} | {'Calc (cm)':<9} | {'Abs Err':<8} | {'Status'}")
    print("-" * 75)

    start_time = time.time()

    for name, w_cm, d_cm, lord_cm, p_val in test_cases:
        gt, res = run_single_adversarial_experiment(simulator, name, w_cm, d_cm, lord_cm, p_val)
        results.append((name, gt, res))
        errors.append(res.absolute_error_cm)

        status = "PASSED" if res.passed_target_0_5cm else "FAILED"
        print(
            f"{name:<30} | {gt.exact_perimeter_cm:<8.2f} | "
            f"{res.calculated_perimeter_cm:<9.2f} | {res.absolute_error_cm:<7.3f} cm | {status}"
        )

    elapsed = time.time() - start_time
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    min_err = float(np.min(errors))

    print("-" * 75)
    print_banner("Benchmark Summary & Statistical Verification")
    print(f"Total Test Cases Evaluated : {len(test_cases)}")
    print(f"Total Frames Processed     : {len(test_cases) * 120} frames (all purged from memory)")
    print(f"Mean Absolute Error        : {mean_err:.3f} cm")
    print(f"Min Absolute Error         : {min_err:.3f} cm")
    print(f"Max Absolute Error         : {max_err:.3f} cm (Target: < 0.500 cm)")
    print(f"Pass Rate                  : {(sum(1 for e in errors if e < 0.5) / len(errors)) * 100.0:.1f}%")
    print(f"Total Execution Time       : {elapsed:.2f} s ({elapsed / (len(test_cases) * 120) * 1000:.1f} ms/frame)")

    # Comparison with naive ellipse model
    print("\n" + "=" * 75)
    print("  COMPARISON: NAIVE ELLIPSE VS ANTHROPOMETRIC LORDOSIS SPLINE")
    print("=" * 75)
    print(f"{'Somatotype':<30} | {'GT (cm)':<8} | {'Naive Ellipse':<14} | {'Spline Model':<14}")
    print("-" * 75)

    for name, gt, res in results:
        a = gt.width_front_cm / 2.0
        b = gt.depth_right_cm / 2.0
        # Ramanujan's formula for naive ellipse
        h = ((a - b) / (a + b)) ** 2
        p_ellipse = np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(4.0 - 3.0 * h)))
        ellipse_err = abs(p_ellipse - gt.exact_perimeter_cm)

        print(
            f"{name:<30} | {gt.exact_perimeter_cm:<8.2f} | "
            f"{p_ellipse:<7.2f} (err {ellipse_err:.2f}) | "
            f"{res.calculated_perimeter_cm:<7.2f} (err {res.absolute_error_cm:.2f})"
        )

    print("-" * 75)
    if max_err < 0.50:
        print("\n>>> ALL ADVERSARIAL QA TESTS PASSED! Target Error < 0.5 cm ACHIEVED. <<<")
        return 0
    else:
        print("\n>>> TARGET FAILED. Max error exceeds 0.5 cm. <<<\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
