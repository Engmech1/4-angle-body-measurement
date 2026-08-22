"""
Exercise App - Live Computer Webcam Capture & Real-Time ML Analysis Engine.

Connects to computer's webcam (or provides synthetic fallback if camera unavailable)
and guides the user through real-time 4-angle guided capture:
1. Real-Time ArUco Marker Detection & Live PPM Overlay
2. Real-Time MediaPipe Pose Skeleton & Anatomical Slice Y-Tracking
3. 30-Frame In-Memory Burst Capture (0, 90, 180, 270 deg) with Zero-Raw-Media Privacy
4. Machine Learning Biomechanical Optimization & Residual Bias Correction
5. Online Active Learning (Press 'U' to input reference tape measure and train ML model live!)
"""

from enum import Enum
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from body_measurement.burst_processor import BurstAngleResult, BurstFrameProcessor
from body_measurement.edge_detection import SubPixelEdgeDetector
from body_measurement.landmarks import AnatomicalAnchorEngine, BodySite
from body_measurement.ml_optimizer import (
    AdaptiveMLReconstructor,
    BiomechanicalFeatureVector,
    MLMeasurementResult,
)
from body_measurement.reconstruction import CrossSectionReconstructor, ReconstructionMethod
from body_measurement.scaling import ArucoMetricScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CaptureState(str, Enum):
    CALIBRATING = "CALIBRATING"
    READY_FRONT_0 = "READY_FRONT_0"
    BURST_FRONT_0 = "BURST_FRONT_0"
    READY_RIGHT_90 = "READY_RIGHT_90"
    BURST_RIGHT_90 = "BURST_RIGHT_90"
    READY_BACK_180 = "READY_BACK_180"
    BURST_BACK_180 = "BURST_BACK_180"
    READY_LEFT_270 = "READY_LEFT_270"
    BURST_LEFT_270 = "BURST_LEFT_270"
    ANALYZING = "ANALYZING"
    SHOW_RESULTS = "SHOW_RESULTS"


class LiveCameraApp:
    """
    Live Interactive Body Measurement Application with Computer Webcam & ML Engine.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_synthetic_camera = False

        # Core CV & ML Modules
        self.scaler = ArucoMetricScaler(marker_size_cm=15.0)
        self.anchor_engine = AnatomicalAnchorEngine()
        self.edge_detector = SubPixelEdgeDetector(gaussian_sigma=1.8, strip_half_height=2)
        self.burst_processor = BurstFrameProcessor(edge_detector=self.edge_detector, mad_threshold=2.5)
        self.ml_optimizer = AdaptiveMLReconstructor()

        # Session State
        self.state = CaptureState.CALIBRATING
        self.selected_site = BodySite.WAIST
        self.pixels_per_cm = 12.5  # Default calibration (updated when ArUco detected)
        self.calibration_verified = False
        self.current_slice_y = None
        self.center_x_hint = None
        self.torso_height_px = 750.0

        # Burst Data across 4 angles
        self.burst_results: Dict[int, BurstAngleResult] = {}
        self.burst_buffer: List[np.ndarray] = []
        self.burst_target_count = 30
        self.last_result: Optional[MLMeasurementResult] = None

        self._init_camera()

    def _init_camera(self) -> None:
        """Initializes OpenCV VideoCapture with fallback to synthetic stream."""
        logger.info(f"Connecting to Camera Index {self.camera_index}...")
        try:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    logger.info("Live Computer Webcam connected successfully!")
                    self.is_synthetic_camera = False
                    return
        except Exception as e:
            logger.warning(f"Error opening camera {self.camera_index}: {e}")

        logger.warning("No physical webcam detected. Enabling Interactive Virtual Camera Simulator.")
        self.is_synthetic_camera = True

    def toggle_camera_mode(self) -> None:
        """Toggles between physical webcam and synthetic simulator."""
        if not self.is_synthetic_camera:
            self.is_synthetic_camera = True
            logger.info("Switched to Synthetic Camera Simulator mode.")
        else:
            self._init_camera()

    def get_frame(self) -> np.ndarray:
        """Fetches frame from webcam or generates synthetic video feed."""
        if not self.is_synthetic_camera and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame

        # Synthetic Interactive Frame Generator
        h, w = 720, 1280
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        
        # Draw background wall & floor line
        cv2.line(frame, (0, int(h * 0.85)), (w, int(h * 0.85)), (80, 80, 80), 2)

        # Draw ArUco marker on wall (top-left)
        dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_px = 150
        if hasattr(cv2.aruco, "generateImageMarker"):
            m_img = cv2.aruco.generateImageMarker(dict_aruco, 0, marker_px, 1)
        else:
            m_img = cv2.aruco.drawMarker(dict_aruco, 0, marker_px, 1)
        m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
        frame[50:50+marker_px, 60:60+marker_px] = m_bgr

        # Draw standing virtual human silhouette in center
        cx = w // 2
        cy = int(h * 0.50)
        t = time.time()
        sway = int(15.0 * np.sin(2.0 * np.pi * 0.4 * t))

        # Dynamic body widths based on angle state
        if "90" in self.state.value or "270" in self.state.value:
            torso_w = 110  # Profile depth
        else:
            torso_w = 175  # Frontal width

        # Body parts
        cv2.circle(frame, (cx + sway, cy - 200), 45, (180, 180, 180), -1)  # Head
        cv2.rectangle(frame, (cx + sway - torso_w, cy - 140), (cx + sway + torso_w, cy + 120), (140, 140, 140), -1)  # Torso
        cv2.rectangle(frame, (cx + sway - torso_w + 10, cy + 120), (cx + sway - 10, cy + 300), (110, 110, 110), -1)  # Left Leg
        cv2.rectangle(frame, (cx + sway + 10, cy + 120), (cx + sway + torso_w - 10, cy + 300), (110, 110, 110), -1)  # Right Leg

        return frame

    def run(self) -> None:
        """Main real-time application loop."""
        window_name = "Exercise App - 4-Angle Guided Capture [Live Camera + ML Engine]"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        fps_timer = time.time()
        frame_count = 0
        fps = 30.0

        logger.info("Interactive Camera Application Started.")
        logger.info("Controls: [SPACE] Capture/Next | [C] Calibrate | [S] Switch Site | [T] Toggle Sim | [U] Update ML | [Q] Quit")

        while True:
            raw_frame = self.get_frame()
            display_frame = raw_frame.copy()
            h, w = display_frame.shape[:2]

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / max(1e-3, (time.time() - fps_timer))
                frame_count = 0
                fps_timer = time.time()

            # 1. Real-Time ArUco Detection & Metric Scaling
            calib_res = self.scaler.detect_and_calibrate(raw_frame)
            if calib_res.is_valid:
                self.pixels_per_cm = calib_res.pixels_per_cm
                self.calibration_verified = True
                pts = calib_res.corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                cv2.putText(display_frame, f"ArUco #{calib_res.marker_id} ({self.pixels_per_cm:.1f} px/cm)",
                            (pts[0][0][0], pts[0][0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2. Real-Time MediaPipe Pose Skeleton & Anchoring
            anchor_res = self.anchor_engine.compute_anchor_slice(raw_frame, site=self.selected_site)
            self.current_slice_y = anchor_res.slice_y_pixel
            self.torso_height_px = anchor_res.torso_height_pixels

            # Compute center of body hint
            if anchor_res.keypoints_summary:
                kp = anchor_res.keypoints_summary
                if "left_hip" in kp and "right_hip" in kp:
                    self.center_x_hint = (kp["left_hip"][0] + kp["right_hip"][0]) / 2.0
                elif "left_shoulder" in kp and "right_shoulder" in kp:
                    self.center_x_hint = (kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2.0

                # Draw skeleton points
                for name, pt in kp.items():
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)

            # Draw anatomical measurement line
            y_line = anchor_res.slice_y_pixel
            cv2.line(display_frame, (50, y_line), (w - 50, y_line), (0, 255, 255), 2)
            cv2.putText(display_frame, f">> {self.selected_site.value.upper()} SLICE (Y: {y_line}px) <<",
                        (60, y_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # 3. State Machine & HUD Rendering
            self._handle_state_machine(raw_frame)
            self._render_hud_overlay(display_frame, fps)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE: Action / Next Step
                self._advance_step()
            elif key == ord('c') or key == ord('C'):  # C: Manual Calibrate
                self.pixels_per_cm = 12.5
                self.calibration_verified = True
                logger.info("Manual calibration scale set: 12.50 px/cm")
            elif key == ord('s') or key == ord('S'):  # S: Toggle Site
                sites = [BodySite.WAIST, BodySite.CHEST, BodySite.HIPS]
                idx = (sites.index(self.selected_site) + 1) % len(sites)
                self.selected_site = sites[idx]
                logger.info(f"Target anatomical site switched to: {self.selected_site.value}")
            elif key == ord('t') or key == ord('T'):  # T: Toggle Simulator
                self.toggle_camera_mode()
            elif key == ord('r') or key == ord('R'):  # R: Reset
                self.state = CaptureState.CALIBRATING
                self.burst_results.clear()
                self.burst_buffer.clear()
                self.last_result = None
                logger.info("Session reset.")
            elif key == ord('u') or key == ord('U'):  # U: Online ML Calibration
                self._prompt_online_ml_update()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Live Camera Application Closed.")

    def _advance_step(self) -> None:
        """Handles user pressing SPACE bar to advance guided steps."""
        if self.state == CaptureState.CALIBRATING:
            self.state = CaptureState.READY_FRONT_0
        elif self.state == CaptureState.READY_FRONT_0:
            self.state = CaptureState.BURST_FRONT_0
            self.burst_buffer.clear()
        elif self.state == CaptureState.READY_RIGHT_90:
            self.state = CaptureState.BURST_RIGHT_90
            self.burst_buffer.clear()
        elif self.state == CaptureState.READY_BACK_180:
            self.state = CaptureState.BURST_BACK_180
            self.burst_buffer.clear()
        elif self.state == CaptureState.READY_LEFT_270:
            self.state = CaptureState.BURST_LEFT_270
            self.burst_buffer.clear()
        elif self.state == CaptureState.SHOW_RESULTS:
            self.state = CaptureState.READY_FRONT_0
            self.burst_results.clear()
            self.burst_buffer.clear()
            self.last_result = None

    def _handle_state_machine(self, frame: np.ndarray) -> None:
        """Collects 30-frame in-memory bursts and runs ML analysis."""
        if self.state in (
            CaptureState.BURST_FRONT_0,
            CaptureState.BURST_RIGHT_90,
            CaptureState.BURST_BACK_180,
            CaptureState.BURST_LEFT_270,
        ):
            self.burst_buffer.append(frame)
            if len(self.burst_buffer) >= self.burst_target_count:
                angle_map = {
                    CaptureState.BURST_FRONT_0: (0, CaptureState.READY_RIGHT_90),
                    CaptureState.BURST_RIGHT_90: (90, CaptureState.READY_BACK_180),
                    CaptureState.BURST_BACK_180: (180, CaptureState.READY_LEFT_270),
                    CaptureState.BURST_LEFT_270: (270, CaptureState.ANALYZING),
                }
                angle_deg, next_state = angle_map[self.state]

                # Process 30 frames in-memory
                res = self.burst_processor.process_burst(
                    frames=self.burst_buffer,
                    y_slice=self.current_slice_y or (frame.shape[0] // 2),
                    angle_degrees=angle_deg,
                    pixels_per_cm=self.pixels_per_cm,
                    center_x_hint=self.center_x_hint,
                )
                self.burst_results[angle_deg] = res
                self.burst_buffer.clear()
                self.state = next_state
                logger.info(f"Angle {angle_deg} deg burst processed: Width = {res.width_cm:.2f} cm (Sway: {res.center_sway_cm:.2f} cm)")

        if self.state == CaptureState.ANALYZING:
            logger.info("Executing Machine Learning Biomechanical Optimization...")
            features = self.ml_optimizer.extract_features(
                burst_data=self.burst_results,
                torso_height_px=self.torso_height_px,
                pixels_per_cm=self.pixels_per_cm,
            )
            ml_res = self.ml_optimizer.predict_and_optimize(features, site=self.selected_site)
            self.last_result = ml_res
            self.state = CaptureState.SHOW_RESULTS
            logger.info(
                f"ML Optimization Complete: Perimeter = {ml_res.ml_corrected_perimeter_cm:.2f} cm "
                f"(p* = {ml_res.adaptive_superellipse_p:.2f}, Bias: {ml_res.predicted_residual_bias_cm:+.3f} cm)"
            )

    def _prompt_online_ml_update(self) -> None:
        """Allows live user calibration check to train ML model online."""
        if self.last_result is None:
            print("\n[ML ONLINE UPDATE] No measurement session available. Complete a 4-angle capture first.")
            return

        print("\n" + "=" * 60)
        print("  ONLINE ML MODEL CALIBRATION & ACTIVE LEARNING")
        print("=" * 60)
        print(f"  Current ML-Estimated Perimeter : {self.last_result.ml_corrected_perimeter_cm:.2f} cm")
        print(f"  Baseline Physical Spline       : {self.last_result.baseline_perimeter_cm:.2f} cm")
        print("-" * 60)
        user_input = input("Enter actual physical tape measurement (cm) or press Enter to cancel: ").strip()

        if user_input:
            try:
                gt_val = float(user_input)
                if 40.0 <= gt_val <= 180.0:
                    err = self.ml_optimizer.online_update(
                        features=self.last_result.features,
                        ground_truth_perimeter_cm=gt_val,
                    )
                    print(f"\n[SUCCESS] ML Model trained online! Calibrated residual bias reduced by: {err:.3f} cm.")
                    self.last_result = self.ml_optimizer.predict_and_optimize(
                        self.last_result.features, site=self.selected_site
                    )
                else:
                    print("[ERROR] Measurement out of plausible range (40 - 180 cm).")
            except ValueError:
                print("[ERROR] Invalid numerical input.")
        print("=" * 60 + "\n")

    def _render_hud_overlay(self, frame: np.ndarray, fps: float) -> None:
        """Renders heads-up display telemetry, instructions, and results."""
        h, w = frame.shape[:2]

        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(frame, "EXERCISE APP: 4-ANGLE GUIDED CAPTURE + ML ENGINE",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        
        status_color = (0, 255, 0) if not self.is_synthetic_camera else (0, 165, 255)
        src_label = "LIVE WEBCAM" if not self.is_synthetic_camera else "SIMULATOR"
        cv2.putText(frame, f"[{src_label}] {fps:.1f} FPS | PPM: {self.pixels_per_cm:.1f}",
                    (w - 320, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

        # Guidance Banner based on State
        state_prompts = {
            CaptureState.CALIBRATING: ("STEP 1: METRIC CALIBRATION", "Show ArUco marker to camera, or press [C] for default scale", (0, 215, 255)),
            CaptureState.READY_FRONT_0: ("ANGLE 1/4: FRONT VIEW (0 deg)", "Stand facing camera. Press [SPACE] to capture burst", (0, 255, 0)),
            CaptureState.BURST_FRONT_0: ("CAPTURING FRONT BURST...", f"Hold still! {len(self.burst_buffer)}/30 frames", (0, 165, 255)),
            CaptureState.READY_RIGHT_90: ("ANGLE 2/4: RIGHT PROFILE (90 deg)", "Turn 90 deg right. Press [SPACE] to capture burst", (0, 255, 0)),
            CaptureState.BURST_RIGHT_90: ("CAPTURING RIGHT BURST...", f"Hold still! {len(self.burst_buffer)}/30 frames", (0, 165, 255)),
            CaptureState.READY_BACK_180: ("ANGLE 3/4: BACK VIEW (180 deg)", "Turn 180 deg back. Press [SPACE] to capture burst", (0, 255, 0)),
            CaptureState.BURST_BACK_180: ("CAPTURING BACK BURST...", f"Hold still! {len(self.burst_buffer)}/30 frames", (0, 165, 255)),
            CaptureState.READY_LEFT_270: ("ANGLE 4/4: LEFT PROFILE (270 deg)", "Turn 270 deg left. Press [SPACE] to capture burst", (0, 255, 0)),
            CaptureState.BURST_LEFT_270: ("CAPTURING LEFT BURST...", f"Hold still! {len(self.burst_buffer)}/30 frames", (0, 165, 255)),
            CaptureState.ANALYZING: ("PROCESSING ML OPTIMIZATION...", "Reconstructing non-elliptical cross section...", (255, 100, 0)),
            CaptureState.SHOW_RESULTS: ("MEASUREMENT COMPLETE!", "Press [U] to calibrate with tape | [SPACE] New Session", (0, 255, 128)),
        }

        title, subtitle, color = state_prompts[self.state]

        # Draw State Banner Box
        cv2.rectangle(frame, (20, h - 85), (w - 20, h - 15), (25, 25, 25), -1)
        cv2.rectangle(frame, (20, h - 85), (w - 20, h - 15), color, 2)
        cv2.putText(frame, title, (40, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        cv2.putText(frame, subtitle, (40, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # If Results Available: Render Biometric Floating Card
        if self.state == CaptureState.SHOW_RESULTS and self.last_result:
            self._render_results_card(frame)

    def _render_results_card(self, frame: np.ndarray) -> None:
        """Renders comprehensive measurement summary card."""
        res = self.last_result
        if not res:
            return

        h, w = frame.shape[:2]
        card_w, card_h = 380, 250
        x1, y1 = w - card_w - 30, 75
        x2, y2 = x1 + card_w, y1 + card_h

        # Translucent background
        sub_img = frame[y1:y2, x1:x2]
        dark_rect = np.full(sub_img.shape, (15, 23, 42), dtype=np.uint8)
        res_blend = cv2.addWeighted(sub_img, 0.15, dark_rect, 0.85, 0)
        frame[y1:y2, x1:x2] = res_blend
        cv2.rectangle(frame, (x1, y1), (x2, y2), (56, 189, 248), 2)

        # Text Header
        cv2.putText(frame, f"{self.selected_site.value.upper()} PERIMETER", (x1 + 15, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (56, 189, 248), 2)

        # Big Number
        cv2.putText(frame, f"{res.ml_corrected_perimeter_cm:.2f} cm", (x1 + 15, y1 + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (74, 222, 128), 3)

        # Sub-metrics
        lines = [
            f"Baseline Spline : {res.baseline_perimeter_cm:.2f} cm",
            f"ML Bias Calib   : {res.predicted_residual_bias_cm:+.2f} cm (p*={res.adaptive_superellipse_p:.2f})",
            f"Uncertainty (95%): +/- {res.estimated_uncertainty_cm:.2f} cm",
            f"Coronal Width   : {res.cross_section_result.coronal_width_cm:.1f} cm",
            f"Sagittal Depth  : {res.cross_section_result.sagittal_depth_cm:.1f} cm",
            f"Cross-Sect Area : {res.cross_section_result.cross_sectional_area_cm2:.0f} cm^2",
        ]

        for i, text in enumerate(lines):
            cv2.putText(frame, text, (x1 + 15, y1 + 110 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (203, 213, 225), 1)


def main():
    app = LiveCameraApp(camera_index=0)
    app.run()


if __name__ == "__main__":
    main()
