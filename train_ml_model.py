"""
Exercise App - Machine Learning Model Trainer & Cross-Validation Benchmark.

Generates a diverse synthetic training dataset across 500 anthropometric somatotypes,
trains the AdaptiveMLReconstructor to learn optimal non-elliptical parameters and
residual bias corrections, and serializes the learned weights to body_measurement/ml_weights.json.
"""

import sys
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from body_measurement.adversarial_simulator import (
    AdversarialSimulationConfig,
    AdversarialSimulator,
)
from body_measurement.burst_processor import BurstFrameProcessor
from body_measurement.edge_detection import SubPixelEdgeDetector
from body_measurement.landmarks import BodySite
from body_measurement.ml_optimizer import (
    AdaptiveMLReconstructor,
    BiomechanicalFeatureVector,
)
from body_measurement.reconstruction import (
    CrossSectionReconstructor,
    ReconstructionMethod,
)


def generate_ml_training_dataset(num_samples: int = 250):
    print(f"\n[STEP 1/3] Generating {num_samples} Diverse Synthetic Biomechanical Profiles...")
    simulator = AdversarialSimulator(AdversarialSimulationConfig(frames_per_angle=15))
    burst_processor = BurstFrameProcessor(edge_detector=SubPixelEdgeDetector(strip_half_height=2))

    X_list = []
    y_residual_list = []
    y_p_list = []
    gt_perimeters = []
    baseline_perimeters = []

    reconstructor = CrossSectionReconstructor()

    start_time = time.time()
    for i in range(num_samples):
        # Randomize anthropometric parameters (Slender -> Obese, Short -> Tall)
        width_cm = np.random.uniform(24.0, 42.0)
        depth_cm = np.random.uniform(16.0, 32.0)
        lordosis_cm = np.random.uniform(1.2, 3.8)
        p_val = np.random.uniform(2.30, 2.62)

        gt = simulator.generate_ground_truth_anatomy(
            nominal_width_cm=width_cm,
            nominal_depth_cm=depth_cm,
            lordosis_depth_cm=lordosis_cm,
            superellipse_p=p_val,
        )

        # Generate 4-angle burst measurements
        burst_data = {}
        ppm = 12.5
        y_slice = 960

        for angle in [0, 90, 180, 270]:
            frames = simulator.generate_adversarial_test_case(gt, angle, inject_occlusion=False)
            res = burst_processor.process_burst(frames, y_slice, angle, ppm)
            burst_data[angle] = res

        # Extract features
        w_0 = burst_data[0].width_cm
        w_180 = burst_data[180].width_cm
        d_90 = burst_data[90].width_cm
        d_270 = burst_data[270].width_cm

        feat = BiomechanicalFeatureVector(
            coronal_width_front_cm=w_0,
            coronal_width_back_cm=w_180,
            sagittal_depth_right_cm=d_90,
            sagittal_depth_left_cm=d_270,
            aspect_ratio=float((w_0 + w_180) / (d_90 + d_270)),
            width_asymmetry_cm=abs(w_0 - w_180),
            depth_asymmetry_cm=abs(d_90 - d_270),
            mean_sway_amplitude_cm=float(np.mean([b.center_sway_cm for b in burst_data.values()])),
            max_sway_amplitude_cm=float(np.max([b.center_sway_cm for b in burst_data.values()])),
            mean_edge_confidence=float(np.mean([b.mean_confidence for b in burst_data.values()])),
            min_edge_confidence=float(np.min([b.mean_confidence for b in burst_data.values()])),
            torso_height_proxy_cm=60.0,
            base_perimeter_estimate_cm=float(np.pi * (w_0 + d_90) / 2.0),
        )

        # Base reconstruction with default adaptive p
        base_res = reconstructor.reconstruct_cross_section(
            width_front_cm=w_0,
            depth_right_cm=d_90,
            width_back_cm=w_180,
            depth_left_cm=d_270,
            custom_lordosis_depth_cm=lordosis_cm,
        )

        residual = gt.exact_perimeter_cm - base_res.perimeter_cm

        X_list.append(feat.to_array())
        y_residual_list.append(residual)
        y_p_list.append(p_val)
        gt_perimeters.append(gt.exact_perimeter_cm)
        baseline_perimeters.append(base_res.perimeter_cm)

        if (i + 1) % 50 == 0:
            print(f"  -> Processed {i+1}/{num_samples} samples...")

    elapsed = time.time() - start_time
    print(f"  -> Dataset generation complete in {elapsed:.2f} s ({elapsed/num_samples*1000:.1f} ms/sample).")

    return (
        np.array(X_list),
        np.array(y_residual_list),
        np.array(y_p_list),
        np.array(gt_perimeters),
        np.array(baseline_perimeters),
    )


def main():
    print("=" * 72)
    print("   EXERCISE APP: ADAPTIVE ML MODEL TRAINING & BENCHMARK")
    print("   Learns Anthropometric Residual Biases & Morphological Superellipse p*")
    print("=" * 72)

    X, y_res, y_p, gt_p, base_p = generate_ml_training_dataset(num_samples=150)

    # Train / Test Split (80% Train, 20% Test)
    print("\n[STEP 2/3] Performing 80/20 Train-Test Split and Training Pipeline...")
    X_train, X_test, y_res_train, y_res_test, y_p_train, y_p_test, gt_train, gt_test, base_train, base_test = (
        train_test_split(
            X, y_res, y_p, gt_p, base_p, test_size=0.20, random_state=42
        )
    )

    ml_optimizer = AdaptiveMLReconstructor()

    # Fit Scaler and Models on Training Data Only (Strict ML Practice)
    ml_optimizer._scaler.fit(X_train)
    X_train_scaled = ml_optimizer._scaler.transform(X_train)
    X_test_scaled = ml_optimizer._scaler.transform(X_test)

    ml_optimizer._residual_model.fit(X_train_scaled, y_res_train)
    ml_optimizer._p_model.fit(X_train_scaled, y_p_train)

    # Evaluate on Unseen Test Data
    print("\n[STEP 3/3] Evaluating ML Performance on Unseen Test Data...")
    pred_res_test = ml_optimizer._residual_model.predict(X_test_scaled)
    pred_p_test = ml_optimizer._p_model.predict(X_test_scaled)

    # Baseline vs ML-Corrected Perimeters on Test Set
    ml_corrected_test = base_test + pred_res_test
    baseline_mae = mean_absolute_error(gt_test, base_test)
    ml_mae = mean_absolute_error(gt_test, ml_corrected_test)
    baseline_max_err = float(np.max(np.abs(gt_test - base_test)))
    ml_max_err = float(np.max(np.abs(gt_test - ml_corrected_test)))

    p_mae = mean_absolute_error(y_p_test, pred_p_test)
    p_r2 = r2_score(y_p_test, pred_p_test)

    # Save trained weights
    ml_optimizer.save_model(ml_optimizer.model_path)

    print("\n" + "=" * 72)
    print("                     ML MODEL EVALUATION REPORT")
    print("=" * 72)
    print(f"  Test Set Size              : {len(X_test)} unseen subjects")
    print(f"  Baseline Algorithm MAE     : {baseline_mae:.4f} cm (Max Err: {baseline_max_err:.4f} cm)")
    print(f"  ML-Augmented Algorithm MAE : {ml_mae:.4f} cm (Max Err: {ml_max_err:.4f} cm)")
    print(f"  Error Reduction Ratio      : {((baseline_mae - ml_mae) / baseline_mae) * 100.0:.1f} % improvement")
    print(f"  Morphology p* Prediction R² : {p_r2:.3f} (MAE: {p_mae:.4f})")
    print(f"  Target Accuracy (< 0.5 cm) : {'PASSED 100%' if ml_max_err < 0.5 else 'REVIEW'}")
    print(f"  Model Weights Serialized   : {ml_optimizer.model_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
