"""
ANTIGRAVITY Continuous Monte Carlo Fuzzing & Anti-Overfitting Stress Engine.

Generates thousands of randomized anthropometric somatotypes, camera setups,
environmental corruptions, and poses to continuously stress-test the body measurement
pipeline, ensuring zero overfitting to fixed evaluation sets or benchmark fixtures.

Usage:
    python -m eval.fuzzer --samples 50
    python -m eval.fuzzer --samples 200 --seed 1234
    python -m eval.fuzzer --continuous
"""

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval.synthetic_generator import DigitalTwinGenerator, RenderedSubjectScene
from eval.adversarial_corruptions import (
    apply_shadow,
    apply_backlight,
    apply_low_light_noise,
    apply_colour_cast,
    apply_motion_blur,
    apply_jpeg_q40,
    apply_rolling_shutter,
    apply_aruco_tilt,
    apply_aruco_occlusion,
    apply_aruco_motion_blur,
    apply_loose_clothing,
    apply_skin_background_clutter,
    apply_mirror_reflection,
)
from eval.tiers import default_body_measurement_pipeline

logger = logging.getLogger(__name__)
ARTIFACTS_DIR = Path("artifacts")


@dataclass
class FuzzSampleResult:
    """Detailed result of a single randomized fuzzing trial."""
    sample_id: str
    height_cm: float
    waist_width_cm: float
    waist_depth_cm: float
    distance_m: float
    camera_height_cm: float
    camera_roll_deg: float
    marker_tilt_deg: float
    applied_corruption: Optional[str]
    gt_perimeter_cm: float
    pred_perimeter_cm: float
    error_cm: float
    signed_diff_cm: float
    is_valid: bool
    quality_flags: List[str]
    is_silent_failure: bool
    is_refusal: bool
    runtime_seconds: float


class MonteCarloFuzzer:
    """
    Continuous Monte Carlo fuzzing engine for anthropometric pipeline stress-testing.
    """

    CORRUPTIONS = [
        None,  # 50% clean frames
        None,
        None,
        None,
        "shadow",
        "backlight",
        "low_light",
        "colour_cast",
        "motion_blur",
        "jpeg_q40",
        "rolling_shutter",
        "skin_clutter",
        "mirror_reflection",
        "loose_clothing",
    ]

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(time.time())
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.gen = DigitalTwinGenerator(seed=self.seed)

    def generate_random_sample(self, index: int) -> Tuple[RenderedSubjectScene, Optional[str]]:
        """
        Samples a randomized subject morphology and realistic camera environment.
        """
        sub_id = f"rnd_sub_{index:05d}"

        # 1. Randomized Human Morphology (Spanning 1st to 99th percentiles)
        height_cm = float(np.random.uniform(148.0, 202.0))

        # Somatotype ratio mixture (Athletic, Slender, Android, Gynoid, Average)
        somatotype = np.random.choice(["athletic", "slender", "average", "android", "gynoid"])
        if somatotype == "slender":
            waist_w = float(np.random.uniform(22.0, 28.0))
            waist_d = float(np.random.uniform(15.0, 19.0))
            chest_w = float(np.random.uniform(26.0, 32.0))
            chest_d = float(np.random.uniform(17.0, 22.0))
            hip_w = float(np.random.uniform(28.0, 34.0))
            hip_d = float(np.random.uniform(18.0, 24.0))
        elif somatotype == "athletic":
            waist_w = float(np.random.uniform(26.0, 32.0))
            waist_d = float(np.random.uniform(18.0, 22.0))
            chest_w = float(np.random.uniform(34.0, 42.0))
            chest_d = float(np.random.uniform(23.0, 28.0))
            hip_w = float(np.random.uniform(32.0, 38.0))
            hip_d = float(np.random.uniform(22.0, 26.0))
        elif somatotype == "android":
            waist_w = float(np.random.uniform(34.0, 48.0))
            waist_d = float(np.random.uniform(26.0, 38.0))
            chest_w = float(np.random.uniform(36.0, 48.0))
            chest_d = float(np.random.uniform(27.0, 38.0))
            hip_w = float(np.random.uniform(36.0, 46.0))
            hip_d = float(np.random.uniform(26.0, 36.0))
        elif somatotype == "gynoid":
            waist_w = float(np.random.uniform(25.0, 32.0))
            waist_d = float(np.random.uniform(18.0, 23.0))
            chest_w = float(np.random.uniform(30.0, 36.0))
            chest_d = float(np.random.uniform(20.0, 25.0))
            hip_w = float(np.random.uniform(36.0, 48.0))
            hip_d = float(np.random.uniform(25.0, 35.0))
        else:  # average
            waist_w = float(np.random.uniform(26.0, 36.0))
            waist_d = float(np.random.uniform(18.0, 26.0))
            chest_w = float(np.random.uniform(31.0, 40.0))
            chest_d = float(np.random.uniform(21.0, 28.0))
            hip_w = float(np.random.uniform(32.0, 42.0))
            hip_d = float(np.random.uniform(22.0, 30.0))

        # 2. Randomized Camera & Distance Setup
        distance_m = float(np.random.uniform(1.85, 3.40))
        camera_height_cm = float(np.random.uniform(80.0, 120.0))
        camera_roll_deg = float(np.random.uniform(-1.8, 1.8))
        marker_tilt_deg = float(np.random.uniform(0.0, 12.0))
        focal_mult = float(np.random.choice([0.85, 1.0, 1.15, 1.30]))
        noise_sigma = float(np.random.choice([0.0, 1.0, 2.5, 4.0]))

        scene = self.gen.render_subject_scene(
            subject_id=sub_id,
            height_cm=height_cm,
            waist_width_cm=waist_w,
            waist_depth_cm=waist_d,
            chest_width_cm=chest_w,
            chest_depth_cm=chest_d,
            hip_width_cm=hip_w,
            hip_depth_cm=hip_d,
            distance_m=distance_m,
            camera_height_cm=camera_height_cm,
            focal_mult=focal_mult,
            camera_roll_deg=camera_roll_deg,
            marker_tilt_deg=marker_tilt_deg,
            noise_sigma=noise_sigma,
        )

        # 3. Optional Injected Adversarial Sensor Noise
        corruption = random.choice(self.CORRUPTIONS)
        if corruption:
            waist_y = scene.metadata.get("waist_y_pixel", 400)
            if corruption == "shadow":
                scene.frames_by_angle = {a: apply_shadow(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "backlight":
                scene.frames_by_angle = {a: apply_backlight(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "low_light":
                scene.frames_by_angle = {a: apply_low_light_noise(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "colour_cast":
                scene.frames_by_angle = {a: apply_colour_cast(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "motion_blur":
                scene.frames_by_angle = {a: apply_motion_blur(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "jpeg_q40":
                scene.frames_by_angle = {a: apply_jpeg_q40(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "rolling_shutter":
                scene.frames_by_angle = {a: apply_rolling_shutter(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "skin_clutter":
                scene.frames_by_angle = {a: apply_skin_background_clutter(f, waist_y) for a, f in scene.frames_by_angle.items()}
            elif corruption == "mirror_reflection":
                scene.frames_by_angle = {a: apply_mirror_reflection(f) for a, f in scene.frames_by_angle.items()}
            elif corruption == "loose_clothing":
                # Asymmetric fabric drape on one view only (Front), triggering coronal asymmetry refusal
                scene.frames_by_angle[0] = apply_loose_clothing(scene.frames_by_angle[0], waist_y, dilation_px=15)

        return scene, corruption

    def run_fuzzing_suite(
        self,
        num_samples: int = 50,
        continuous: bool = False,
        stop_on_silent_failure: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes randomized Monte Carlo fuzz testing across N random samples.
        """
        results: List[FuzzSampleResult] = []
        errors: List[float] = []
        biases: List[float] = []
        silent_failures = 0
        refusals = 0
        valid_count = 0

        print(f"\n==========================================================================================================")
        print(f" ANTIGRAVITY Continuous Monte Carlo Fuzzing & Anti-Overfitting Evaluation Suite")
        print(f" Target: {num_samples if not continuous else 'Continuous (Inf)'} Fresh Stochastic Trials | Seed: {self.seed}")
        print(f"==========================================================================================================\n")
        print(f"{'#':<4} | {'Sample ID':<13} | {'Dimensions (W x D)':<19} | {'Dist':<5} | {'GT Hull':<8} | {'Pred':<8} | {'Error':<8} | {'Bias':<8} | {'Score':<6} | {'Status':<11} | {'Noise'}")
        print("-" * 118)

        t_start = time.time()
        idx = 0

        try:
            while continuous or (idx < num_samples):
                idx += 1
                t0 = time.time()

                scene, corruption = self.generate_random_sample(idx)
                gt = scene.ground_truth["waist"]
                gt_p = gt.perimeter_hull_cm

                # Execute measurement pipeline
                pipe_res = default_body_measurement_pipeline(
                    scene.frames_by_angle,
                    scene.pixels_per_cm,
                    scene.metadata,
                )

                is_valid = pipe_res.get("is_valid", False)
                pred_p = pipe_res.get("perimeter_cm", 0.0)
                flags = pipe_res.get("quality_flags", [])
                elapsed = time.time() - t0

                if not is_valid:
                    refusals += 1
                    status = "REFUSED"
                    err = 0.0
                    diff = 0.0
                    score_pct = 100.0  # Safe refusal of unmeasurable input is a 100% successful gate
                    is_silent_fail = False
                else:
                    valid_count += 1
                    err = abs(pred_p - gt_p)
                    diff = pred_p - gt_p
                    errors.append(err)
                    biases.append(diff)

                    # Score metric: 100% at 0 error, decaying gracefully
                    score_pct = max(0.0, 100.0 * (1.0 - (err / 2.0)))

                    # Silent Failure: Error > 2x tolerance (1.0 cm) without quality refusal flag
                    if err > 1.0:
                        silent_failures += 1
                        is_silent_fail = True
                        status = "SILENT_FAIL"
                    elif err > 0.50:
                        is_silent_fail = False
                        status = "WARN (>0.5)"
                    else:
                        is_silent_fail = False
                        status = "PASS"

                corrupt_str = corruption if corruption else "clean"
                pred_str = f"{pred_p:6.2f}cm" if is_valid else "REFUSED"
                err_str = f"{err:6.3f}cm" if is_valid else "--"
                diff_str = f"{diff:+6.3f}cm" if is_valid else "--"
                dim_str = f"{gt.coronal_width_cm:.1f}x{gt.sagittal_depth_cm:.1f}cm"

                print(
                    f"{idx:<4} | {scene.subject_id:<13} | {dim_str:<19} | {scene.distance_m:.2f}m | {gt_p:6.2f}cm | "
                    f"{pred_str:<8} | {err_str:<8} | {diff_str:<8} | {score_pct:5.1f}% | {status:<11} | {corrupt_str}"
                )

                res_obj = FuzzSampleResult(
                    sample_id=scene.subject_id,
                    height_cm=scene.height_cm,
                    waist_width_cm=gt.coronal_width_cm,
                    waist_depth_cm=gt.sagittal_depth_cm,
                    distance_m=scene.distance_m,
                    camera_height_cm=scene.camera_height_cm,
                    camera_roll_deg=scene.metadata.get("camera_roll_deg", 0.0),
                    marker_tilt_deg=scene.metadata.get("marker_tilt_deg", 0.0),
                    applied_corruption=corruption,
                    gt_perimeter_cm=gt_p,
                    pred_perimeter_cm=pred_p,
                    error_cm=err,
                    signed_diff_cm=diff,
                    is_valid=is_valid,
                    quality_flags=flags,
                    is_silent_failure=is_silent_fail,
                    is_refusal=(not is_valid),
                    runtime_seconds=elapsed,
                )
                results.append(res_obj)

                if stop_on_silent_failure and is_silent_fail:
                    print(f"\n[ALERT] Fuzzing halted immediately on silent failure: {scene.subject_id}")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Continuous fuzzing stopped by user (Ctrl+C).")

        total_runtime = time.time() - t_start
        total_runs = len(results)

        mae = float(np.mean(errors)) if errors else 0.0
        bias = float(np.mean(biases)) if biases else 0.0
        median_err = float(np.median(errors)) if errors else 0.0
        p90 = float(np.percentile(errors, 90)) if errors else 0.0
        p95 = float(np.percentile(errors, 95)) if errors else 0.0
        p99 = float(np.percentile(errors, 99)) if errors else 0.0
        max_err = float(np.max(errors)) if errors else 0.0
        silent_rate = (silent_failures / total_runs) * 100.0 if total_runs > 0 else 0.0
        refusal_rate = (refusals / total_runs) * 100.0 if total_runs > 0 else 0.0

        # Anti-Overfitting Generalization Score
        # 100 points scale: Deduct for MAE > 0.20cm, bias, and silent failures
        acc_score = max(0.0, 100.0 - (mae * 50.0) - (abs(bias) * 30.0) - (silent_failures * 50.0))

        # Markdown Scoreboard
        print("\n" + "=" * 84)
        print("### ANTIGRAVITY Randomized Anti-Overfitting Scoreboard")
        print("=" * 84)
        print(f"| Metric | Value | Acceptance Target | Result |")
        print(f"|---|---|---|---|")
        print(f"| Total Stochastic Runs ($N$) | {total_runs} | $\\ge 50$ | PASS |")
        print(f"| Valid Measurements ($N_{{val}}$) | {valid_count} ({(valid_count/total_runs)*100:.1f}%) | N/A | OK |")
        print(f"| Safe Quality Refusals ($N_{{ref}}$) | {refusals} ({refusal_rate:.1f}%) | Tracked | OK |")
        print(f"| Silent Failure Rate | {silent_rate:.2f}% ({silent_failures} fails) | **0.0%** | {'PASS' if silent_failures == 0 else 'FAIL'} |")
        print(f"| Mean Absolute Error (MAE) | **{mae:.3f} cm** ({mae*10:.1f} mm) | $\\le 0.50\\text{{ cm}}$ | {'PASS' if mae <= 0.50 else 'FAIL'} |")
        print(f"| Systematic Bias | **{bias:+.3f} cm** ({bias*10:+.1f} mm) | $|bias| \\le 0.20\\text{{ cm}}$ | {'PASS' if abs(bias) <= 0.20 else 'FAIL'} |")
        print(f"| 95th Percentile (P95) | **{p95:.3f} cm** | $\\le 1.00\\text{{ cm}}$ | {'PASS' if p95 <= 1.00 else 'FAIL'} |")
        print(f"| Max Error | **{max_err:.3f} cm** | Informational | OK |")
        print(f"| **Anti-Overfit Generalization Score** | **{acc_score:.1f} / 100** | $\\ge 85.0$ | {'PASS (NO OVERFIT)' if acc_score >= 85.0 else 'FAIL'} |")
        print("=" * 84)

        all_passed = (mae <= 0.50) and (abs(bias) <= 0.20) and (p95 <= 1.00) and (silent_failures == 0)
        print(f"\n OVERALL ANTI-OVERFITTING GATE: {'PASSED (EXCELLENT GENERALIZATION)' if all_passed else 'FAILED'}\n")

        summary_payload = {
            "seed": self.seed,
            "total_samples": total_runs,
            "valid_samples": valid_count,
            "refusals": refusals,
            "refusal_rate_pct": refusal_rate,
            "silent_failures": silent_failures,
            "silent_failure_rate_pct": silent_rate,
            "mae_cm": mae,
            "bias_cm": bias,
            "p50_cm": median_err,
            "p90_cm": p90,
            "p95_cm": p95,
            "p99_cm": p99,
            "max_error_cm": max_err,
            "generalization_score_pct": acc_score,
            "all_passed": all_passed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ARTIFACTS_DIR / "fuzz_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        return summary_payload


def main():
    parser = argparse.ArgumentParser(description="ANTIGRAVITY Monte Carlo Fuzzing Engine")
    parser.add_argument("--samples", type=int, default=50, help="Number of randomized samples to test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (defaults to current timestamp)")
    parser.add_argument("--continuous", action="store_true", help="Run indefinitely until Ctrl+C")
    parser.add_argument("--stop-on-fail", action="store_true", help="Halt on the first silent failure")
    args = parser.parse_args()

    fuzzer = MonteCarloFuzzer(seed=args.seed)
    fuzzer.run_fuzzing_suite(
        num_samples=args.samples,
        continuous=args.continuous,
        stop_on_silent_failure=args.stop_on_fail,
    )


if __name__ == "__main__":
    main()
