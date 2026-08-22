"""
Exercise App - Multi-View Calibration Dashboard & Biomechanical Diagnostic Workbench.

A multi-camera/phone-to-OpenCV diagnostic workbench for visual calibration:
- Quadrant 1: Raw Capture + ArUco 3D Pose / PPM Calibration & Camera Tilt Angle
- Quadrant 2: MediaPipe 33-Keypoint Pose Skeleton & Anatomical Slice Anchoring
- Quadrant 3: 1D Sub-Pixel DoG Gradient Oscilloscope & Parabolic Peak Spectrum
- Quadrant 4: 2D Anthropometric Non-Elliptical Cross-Section & ML Biometrics Card

Supports:
- PC Built-in / USB Webcams (Index 0, 1, 2)
- Smartphone IP Cameras (DroidCam, IP Webcam, Iriun, RTSP/HTTP URL)
- Interactive Virtual Simulator Fallback
- Live OpenCV Trackbars for Tuning Parameters in Real-Time
"""

import argparse
from enum import Enum
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from body_measurement.edge_detection import EdgeSliceResult, SubPixelEdgeDetector
from body_measurement.landmarks import AnatomicalAnchorEngine, BodySite
from body_measurement.ml_optimizer import AdaptiveMLReconstructor, BiomechanicalFeatureVector
from body_measurement.reconstruction import CrossSectionReconstructor, ReconstructionMethod
from body_measurement.scaling import ArucoMetricScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ViewMode(str, Enum):
    QUAD_VIEW = "QUAD_VIEW"
    RAW_VIEW = "RAW_VIEW"
    POSE_VIEW = "POSE_VIEW"
    OSCILLOSCOPE_VIEW = "OSCILLOSCOPE_VIEW"
    CROSS_SECTION_VIEW = "CROSS_SECTION_VIEW"


class CalibrationDashboardApp:
    """
    Multi-View Visual Debug & Calibration Workbench.
    """

    def __init__(self, camera_source: Union[int, str] = 0):
        self.camera_source = camera_source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_synthetic_camera = False

        # Core Engines
        self.scaler = ArucoMetricScaler(marker_size_cm=15.0)
        self.anchor_engine = AnatomicalAnchorEngine()
        self.edge_detector = SubPixelEdgeDetector(gaussian_sigma=1.8, strip_half_height=2)
        self.reconstructor = CrossSectionReconstructor()
        self.ml_optimizer = AdaptiveMLReconstructor()

        # Calibration State
        self.pixels_per_cm = 12.5
        self.marker_size_cm = 15.0
        self.selected_site = BodySite.WAIST
        self.site_ratio_override: Optional[float] = None
        self.view_mode = ViewMode.QUAD_VIEW

        # Trackbar tuning values
        self.trackbar_sigma = 18  # 1.8 * 10
        self.trackbar_dog_thresh = 15
        self.trackbar_marker_size = 15
        self.trackbar_site_ratio = 62  # 0.618 * 100

        # Frame Dimensions
        self.target_canvas_w = 1600
        self.target_canvas_h = 900

        self._init_camera()

    def _init_camera(self) -> None:
        """Connects to Webcam, Phone IP Camera stream, or initializes Simulator."""
        logger.info(f"Connecting to Camera Source: '{self.camera_source}'...")
        try:
            if isinstance(self.camera_source, int):
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.camera_source, backend)
            else:
                # Network IP camera / RTSP stream (e.g. http://192.168.1.50:8080/video)
                self.cap = cv2.VideoCapture(str(self.camera_source))

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    logger.info(f"Camera source '{self.camera_source}' connected successfully!")
                    self.is_synthetic_camera = False
                    return
        except Exception as e:
            logger.warning(f"Failed to connect to camera '{self.camera_source}': {e}")

        logger.warning("No physical/network camera available. Activating Virtual Interactive Simulator.")
        self.is_synthetic_camera = True

    def toggle_camera_mode(self) -> None:
        """Toggles between camera feed and synthetic simulator."""
        if not self.is_synthetic_camera:
            self.is_synthetic_camera = True
            logger.info("Switched to Synthetic Simulator mode.")
        else:
            self._init_camera()

    def set_camera_source(self, new_source: Union[int, str]) -> None:
        """Switches camera source URL or index."""
        if self.cap:
            self.cap.release()
        self.camera_source = new_source
        self._init_camera()

    def get_frame(self) -> np.ndarray:
        """Fetches live frame or generates virtual human stream."""
        if not self.is_synthetic_camera and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame

        # Virtual Diagnostic Simulator Canvas
        h, w = 720, 1280
        frame = np.full((h, w, 3), 32, dtype=np.uint8)

        # Floor grid and wall lines
        cv2.line(frame, (0, int(h * 0.85)), (w, int(h * 0.85)), (90, 90, 90), 2)
        for gx in range(0, w, 80):
            cv2.line(frame, (gx, int(h * 0.85)), (int(gx + (gx - w // 2) * 0.5), h), (60, 60, 60), 1)

        # ArUco Marker on wall
        dict_aruco = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_px = 150
        if hasattr(cv2.aruco, "generateImageMarker"):
            m_img = cv2.aruco.generateImageMarker(dict_aruco, 0, marker_px, 1)
        else:
            m_img = cv2.aruco.drawMarker(dict_aruco, 0, marker_px, 1)
        m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
        frame[40:40+marker_px, 60:60+marker_px] = m_bgr

        # Virtual Subject with dynamic sway
        cx = w // 2
        cy = int(h * 0.50)
        t = time.time()
        sway = int(18.0 * np.sin(2.0 * np.pi * 0.35 * t))

        torso_w = 180
        # Head
        cv2.circle(frame, (cx + sway, cy - 210), 45, (190, 190, 190), -1)
        # Neck
        cv2.rectangle(frame, (cx + sway - 15, cy - 170), (cx + sway + 15, cy - 140), (170, 170, 170), -1)
        # Torso
        cv2.rectangle(frame, (cx + sway - torso_w, cy - 140), (cx + sway + torso_w, cy + 120), (150, 150, 150), -1)
        # Left & Right Legs
        cv2.rectangle(frame, (cx + sway - torso_w + 10, cy + 120), (cx + sway - 10, cy + 300), (120, 120, 120), -1)
        cv2.rectangle(frame, (cx + sway + 10, cy + 120), (cx + sway + torso_w - 10, cy + 300), (120, 120, 120), -1)

        return frame

    def run(self) -> None:
        """Main real-time multi-view workbench loop."""
        window_name = "Exercise App - Multi-View Calibration Dashboard & Biomechanical Workbench"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.target_canvas_w, self.target_canvas_h)

        # Create Tuning Trackbars
        cv2.createTrackbar("Sigma x10", window_name, self.trackbar_sigma, 40, lambda v: None)
        cv2.createTrackbar("DoG Thresh", window_name, self.trackbar_dog_thresh, 80, lambda v: None)
        cv2.createTrackbar("Marker cm", window_name, self.trackbar_marker_size, 40, lambda v: None)
        cv2.createTrackbar("Site Ratio %", window_name, self.trackbar_site_ratio, 100, lambda v: None)

        fps_timer = time.time()
        frame_count = 0
        fps = 30.0

        logger.info("Starting Multi-View Calibration Dashboard...")
        logger.info("Controls: [V] Switch Views | [P] Phone IP Camera URL | [S] Switch Site | [T] Toggle Simulator | [Q] Quit")

        while True:
            # Read Trackbars
            self.trackbar_sigma = max(5, cv2.getTrackbarPos("Sigma x10", window_name))
            self.trackbar_dog_thresh = max(2, cv2.getTrackbarPos("DoG Thresh", window_name))
            self.trackbar_marker_size = max(5, cv2.getTrackbarPos("Marker cm", window_name))
            self.trackbar_site_ratio = max(10, cv2.getTrackbarPos("Site Ratio %", window_name))

            # Apply Trackbars
            sigma = self.trackbar_sigma / 10.0
            if abs(self.edge_detector.gaussian_sigma - sigma) > 0.05:
                self.edge_detector = SubPixelEdgeDetector(
                    gaussian_sigma=sigma,
                    min_gradient_threshold=float(self.trackbar_dog_thresh),
                )
            self.scaler.marker_size_cm = float(self.trackbar_marker_size)
            self.site_ratio_override = self.trackbar_site_ratio / 100.0

            raw_frame = self.get_frame()

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / max(1e-3, (time.time() - fps_timer))
                frame_count = 0
                fps_timer = time.time()

            # Build the 4 Diagnostic Panels
            panel1 = self._build_panel_raw_aruco(raw_frame, fps)
            panel2, anchor_res = self._build_panel_pose_anchor(raw_frame)
            panel3, edge_res = self._build_panel_oscilloscope(raw_frame, anchor_res)
            panel4 = self._build_panel_cross_section_ml(edge_res, anchor_res)

            # Compose Master Multi-View Canvas
            if self.view_mode == ViewMode.QUAD_VIEW:
                # 2x2 Grid (1600x900)
                top_row = np.hstack([panel1, panel2])
                bottom_row = np.hstack([panel3, panel4])
                canvas = np.vstack([top_row, bottom_row])
            elif self.view_mode == ViewMode.RAW_VIEW:
                canvas = cv2.resize(panel1, (self.target_canvas_w, self.target_canvas_h))
            elif self.view_mode == ViewMode.POSE_VIEW:
                canvas = cv2.resize(panel2, (self.target_canvas_w, self.target_canvas_h))
            elif self.view_mode == ViewMode.OSCILLOSCOPE_VIEW:
                canvas = cv2.resize(panel3, (self.target_canvas_w, self.target_canvas_h))
            elif self.view_mode == ViewMode.CROSS_SECTION_VIEW:
                canvas = cv2.resize(panel4, (self.target_canvas_w, self.target_canvas_h))

            # Master Top HUD Overlay
            self._draw_master_hud(canvas, fps)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q / ESC
                break
            elif key == ord('v') or key == ord('V'):  # V: Cycle View Modes
                modes = list(ViewMode)
                idx = (modes.index(self.view_mode) + 1) % len(modes)
                self.view_mode = modes[idx]
                logger.info(f"Switched View Mode to: {self.view_mode.value}")
            elif key == ord('s') or key == ord('S'):  # S: Toggle Body Site
                sites = [BodySite.WAIST, BodySite.CHEST, BodySite.HIPS]
                idx = (sites.index(self.selected_site) + 1) % len(sites)
                self.selected_site = sites[idx]
                logger.info(f"Target Site: {self.selected_site.value}")
            elif key == ord('t') or key == ord('T'):  # T: Toggle Simulator
                self.toggle_camera_mode()
            elif key == ord('p') or key == ord('P'):  # P: Connect Phone IP Camera
                self._prompt_phone_camera_url()

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Calibration Dashboard closed.")

    def _prompt_phone_camera_url(self) -> None:
        """Console prompt to enter phone camera RTSP/HTTP URL."""
        print("\n" + "=" * 65)
        print("  CONNECT PHONE CAMERA TO OPENCV (DroidCam / IP Webcam / Iriun)")
        print("=" * 65)
        print("  Examples:")
        print("   - IP Webcam app : http://192.168.1.50:8080/video")
        print("   - DroidCam app  : http://192.168.1.50:4747/video")
        print("   - USB Webcam Id : 0 or 1 or 2")
        print("-" * 65)
        user_url = input("Enter Phone Camera URL or Webcam Index (or Enter to cancel): ").strip()
        if user_url:
            if user_url.isdigit():
                self.set_camera_source(int(user_url))
            else:
                self.set_camera_source(user_url)
        print("=" * 65 + "\n")

    # =========================================================================
    # Panel 1: Raw Capture + ArUco 3D Pose / PPM Calibration (800x450)
    # =========================================================================
    def _build_panel_raw_aruco(self, frame: np.ndarray, fps: float) -> np.ndarray:
        p_w, p_h = 800, 450
        panel = cv2.resize(frame, (p_w, p_h))

        # Detect ArUco Marker
        calib_res = self.scaler.detect_and_calibrate(frame)
        
        # Panel Header
        cv2.rectangle(panel, (0, 0), (p_w, 32), (20, 20, 20), -1)
        cv2.putText(panel, "[1] RAW CAPTURE & ARUCO 3D METRIC CALIBRATION", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2)

        if calib_res.is_valid:
            self.pixels_per_cm = calib_res.pixels_per_cm
            scale_x = p_w / frame.shape[1]
            scale_y = p_h / frame.shape[0]

            pts = (calib_res.corners * np.array([scale_x, scale_y])).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(panel, [pts], True, (74, 222, 128), 2)
            
            # Center of marker
            mc_x = int(np.mean(pts[:, 0, 0]))
            mc_y = int(np.mean(pts[:, 0, 1]))
            cv2.circle(panel, (mc_x, mc_y), 5, (0, 255, 255), -1)

            # Draw Simulated 3D Axis
            cv2.line(panel, (mc_x, mc_y), (mc_x + 35, mc_y), (0, 0, 255), 2)  # X-Axis (Red)
            cv2.line(panel, (mc_x, mc_y), (mc_x, mc_y - 35), (0, 255, 0), 2)  # Y-Axis (Green)
            cv2.line(panel, (mc_x, mc_y), (mc_x - 25, mc_y + 25), (255, 0, 0), 2)  # Z-Axis (Blue)

            # Metric Telemetry Box
            cv2.rectangle(panel, (12, p_h - 70), (280, p_h - 10), (15, 23, 42), -1)
            cv2.rectangle(panel, (12, p_h - 70), (280, p_h - 10), (74, 222, 128), 1)
            cv2.putText(panel, f"PPM: {self.pixels_per_cm:.2f} px/cm", (20, p_h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (74, 222, 128), 1)
            cv2.putText(panel, f"Marker Size: {self.scaler.marker_size_cm:.1f} cm | Id: #{calib_res.marker_id}",
                        (20, p_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1)
        else:
            # Prompt calibration
            cv2.rectangle(panel, (12, p_h - 60), (320, p_h - 10), (15, 23, 42), -1)
            cv2.rectangle(panel, (12, p_h - 60), (320, p_h - 10), (248, 113, 113), 1)
            cv2.putText(panel, "ArUco Not Detected (Using 12.5 px/cm)", (20, p_h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 113, 113), 1)
            cv2.putText(panel, "Show 15cm marker to camera or tune trackbar", (20, p_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1)

        return panel

    # =========================================================================
    # Panel 2: MediaPipe 33-Keypoint Pose & Anatomical Anchoring (800x450)
    # =========================================================================
    def _build_panel_pose_anchor(self, frame: np.ndarray) -> Tuple[np.ndarray, any]:
        p_w, p_h = 800, 450
        panel = cv2.resize(frame, (p_w, p_h))

        # Panel Header
        cv2.rectangle(panel, (0, 0), (p_w, 32), (20, 20, 20), -1)
        cv2.putText(panel, "[2] MEDIAPIPE POSE SKELETON & ANATOMICAL ANCHORING", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 204, 21), 2)

        # Run Anatomical Anchor Engine
        anchor_res = self.anchor_engine.compute_anchor_slice(
            frame,
            site=self.selected_site,
            custom_ratio=self.site_ratio_override,
        )

        scale_x = p_w / frame.shape[1]
        scale_y = p_h / frame.shape[0]

        # Draw Pose Skeleton & Joints
        kp = anchor_res.keypoints_summary
        if kp:
            # Draw shoulder and hip connections
            if "left_shoulder" in kp and "right_shoulder" in kp:
                p1 = (int(kp["left_shoulder"][0] * scale_x), int(kp["left_shoulder"][1] * scale_y))
                p2 = (int(kp["right_shoulder"][0] * scale_x), int(kp["right_shoulder"][1] * scale_y))
                cv2.line(panel, p1, p2, (56, 189, 248), 2)

            if "left_hip" in kp and "right_hip" in kp:
                p1 = (int(kp["left_hip"][0] * scale_x), int(kp["left_hip"][1] * scale_y))
                p2 = (int(kp["right_hip"][0] * scale_x), int(kp["right_hip"][1] * scale_y))
                cv2.line(panel, p1, p2, (56, 189, 248), 2)

            # Torso Spine Line
            if "left_shoulder" in kp and "left_hip" in kp:
                p1 = (int(kp["left_shoulder"][0] * scale_x), int(kp["left_shoulder"][1] * scale_y))
                p2 = (int(kp["left_hip"][0] * scale_x), int(kp["left_hip"][1] * scale_y))
                cv2.line(panel, p1, p2, (148, 163, 184), 1)

            if "right_shoulder" in kp and "right_hip" in kp:
                p1 = (int(kp["right_shoulder"][0] * scale_x), int(kp["right_shoulder"][1] * scale_y))
                p2 = (int(kp["right_hip"][0] * scale_x), int(kp["right_hip"][1] * scale_y))
                cv2.line(panel, p1, p2, (148, 163, 184), 1)

            for name, pt in kp.items():
                cv2.circle(panel, (int(pt[0] * scale_x), int(pt[1] * scale_y)), 5, (250, 204, 21), -1)

        # Draw Active Anatomical Slice Line
        y_slice_panel = int(anchor_res.slice_y_pixel * scale_y)
        cv2.line(panel, (20, y_slice_panel), (p_w - 20, y_slice_panel), (74, 222, 128), 2)
        cv2.putText(panel, f">> {self.selected_site.value.upper()} SLICE (Y: {anchor_res.slice_y_pixel}px | Ratio: {self.site_ratio_override or 0.618:.2f}) <<",
                    (30, y_slice_panel - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (74, 222, 128), 1)

        # Telemetry Box
        cv2.rectangle(panel, (p_w - 260, p_h - 65), (p_w - 12, p_h - 10), (15, 23, 42), -1)
        cv2.rectangle(panel, (p_w - 260, p_h - 65), (p_w - 12, p_h - 10), (250, 204, 21), 1)
        cv2.putText(panel, f"Torso Span : {anchor_res.torso_height_pixels:.0f} px ({anchor_res.torso_height_pixels/self.pixels_per_cm:.1f} cm)",
                    (p_w - 250, p_h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (203, 213, 225), 1)
        cv2.putText(panel, f"Pose Confidence: {anchor_res.confidence*100.0:.0f} %",
                    (p_w - 250, p_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (74, 222, 128), 1)

        return panel, anchor_res

    # =========================================================================
    # Panel 3: 1D Sub-Pixel DoG Oscilloscope & Edge Spectrum (800x450)
    # =========================================================================
    def _build_panel_oscilloscope(self, frame: np.ndarray, anchor_res: any) -> Tuple[np.ndarray, EdgeSliceResult]:
        p_w, p_h = 800, 450
        panel = np.full((p_h, p_w, 3), 15, dtype=np.uint8)

        # Header
        cv2.rectangle(panel, (0, 0), (p_w, 32), (20, 20, 20), -1)
        cv2.putText(panel, "[3] 1D SUB-PIXEL DOG OSCILLOSCOPE & EDGE SPECTRUM", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (244, 114, 182), 2)

        # Compute 1D Profile across measurement line
        y_slice = anchor_res.slice_y_pixel if anchor_res else frame.shape[0] // 2
        h_f, w_f = frame.shape[:2]
        y_slice = int(np.clip(y_slice, 2, h_f - 3))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        strip = gray[y_slice - 2:y_slice + 3, :].astype(np.float64)
        prof_1d = np.mean(strip, axis=0)  # (w_f,)

        # Compute DoG Gradient
        grad_1d = np.convolve(prof_1d, self.edge_detector.dog_kernel, mode="same")
        grad_mag = np.abs(grad_1d)

        # Center hint
        center_x_hint = None
        if anchor_res and anchor_res.keypoints_summary:
            kp = anchor_res.keypoints_summary
            if "left_hip" in kp and "right_hip" in kp:
                center_x_hint = (kp["left_hip"][0] + kp["right_hip"][0]) / 2.0

        # Sub-Pixel Edge Extraction
        edge_res = self.edge_detector.extract_slice_edges(
            frame,
            y_slice=y_slice,
            center_x_hint=center_x_hint,
        )

        # Draw Grid Lines
        for gy in range(50, p_h - 40, 50):
            cv2.line(panel, (40, gy), (p_w - 20, gy), (30, 41, 59), 1)

        # Graph 1: Raw Intensity Profile I(x) in Top Half (Y: 45 - 210)
        y_base_prof = 210
        prof_norm = (prof_1d - np.min(prof_1d)) / (max(1.0, np.max(prof_1d) - np.min(prof_1d)))
        pts_prof = []
        for x_idx, val in enumerate(prof_norm):
            px = int(40 + (x_idx / w_f) * (p_w - 60))
            py = int(y_base_prof - val * 150)
            pts_prof.append([px, py])
        if len(pts_prof) > 1:
            cv2.polylines(panel, [np.array(pts_prof, dtype=np.int32)], False, (148, 163, 184), 1)

        # Graph 2: Gradient Magnitude |DoG| in Bottom Half (Y: 220 - 390)
        y_base_grad = 390
        grad_max = max(1e-3, np.max(grad_mag))
        grad_norm = grad_mag / grad_max
        pts_grad = []
        for x_idx, val in enumerate(grad_norm):
            px = int(40 + (x_idx / w_f) * (p_w - 60))
            py = int(y_base_grad - val * 150)
            pts_grad.append([px, py])
        if len(pts_grad) > 1:
            cv2.polylines(panel, [np.array(pts_grad, dtype=np.int32)], False, (56, 189, 248), 2)

        # Draw Sub-Pixel Peak Marker Lines
        if edge_res.is_valid:
            px_left = int(40 + (edge_res.left_edge_x / w_f) * (p_w - 60))
            px_right = int(40 + (edge_res.right_edge_x / w_f) * (p_w - 60))

            cv2.line(panel, (px_left, 45), (px_left, y_base_grad), (74, 222, 128), 2)
            cv2.putText(panel, f"L: {edge_res.left_edge_x:.2f}px", (px_left - 30, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (74, 222, 128), 1)

            cv2.line(panel, (px_right, 45), (px_right, y_base_grad), (244, 114, 182), 2)
            cv2.putText(panel, f"R: {edge_res.right_edge_x:.2f}px", (px_right - 30, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (244, 114, 182), 1)

            # Draw span arrow between edges
            cv2.arrowedLine(panel, (px_left, 130), (px_right, 130), (250, 204, 21), 2, tipLength=0.03)
            cv2.arrowedLine(panel, (px_right, 130), (px_left, 130), (250, 204, 21), 2, tipLength=0.03)
            width_cm = edge_res.width_pixels / max(0.1, self.pixels_per_cm)
            cv2.putText(panel, f"Width: {edge_res.width_pixels:.1f}px ({width_cm:.2f} cm)",
                        (int((px_left + px_right) / 2) - 65, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (250, 204, 21), 1)

        # Oscilloscope Telemetry Footer
        cv2.rectangle(panel, (12, p_h - 45), (p_w - 12, p_h - 8), (15, 23, 42), -1)
        cv2.rectangle(panel, (12, p_h - 45), (p_w - 12, p_h - 8), (51, 65, 85), 1)
        cv2.putText(panel, f"DoG Sigma: {self.edge_detector.gaussian_sigma:.1f} | DoG Threshold: {self.edge_detector.min_gradient_threshold:.0f} | Edge SNR: {edge_res.confidence*10.0:.1f} dB",
                    (20, p_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1)

        return panel, edge_res

    # =========================================================================
    # Panel 4: 2D Anthropometric Cross-Section & ML Biometrics Card (800x450)
    # =========================================================================
    def _build_panel_cross_section_ml(self, edge_res: EdgeSliceResult, anchor_res: any) -> np.ndarray:
        p_w, p_h = 800, 450
        panel = np.full((p_h, p_w, 3), 15, dtype=np.uint8)

        # Header
        cv2.rectangle(panel, (0, 0), (p_w, 32), (20, 20, 20), -1)
        cv2.putText(panel, "[4] 2D CROSS-SECTION CONTOUR & BIOMETRIC REPORT", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (74, 222, 128), 2)

        # Estimate Frontal Width and Sagittal Depth
        width_front_cm = (edge_res.width_pixels / max(0.1, self.pixels_per_cm)) if edge_res.is_valid else 32.0
        width_front_cm = float(np.clip(width_front_cm, 15.0, 60.0))
        # Morphological aspect ratio proxy ~ 1.44 for waist
        depth_right_cm = float(width_front_cm / 1.44)

        # Reconstruct 2D Lordosis Cross Section
        recon_res = self.reconstructor.reconstruct_cross_section(
            width_front_cm=width_front_cm,
            depth_right_cm=depth_right_cm,
            width_back_cm=width_front_cm,
            depth_left_cm=depth_right_cm,
            method=ReconstructionMethod.ANTHROPOMETRIC_LORDOSIS_SPLINE,
        )

        # Draw 2D Contour in Left Half (Center: 200, 240)
        cx, cy = 200, 240
        scale_plot = 5.8  # px per cm

        # Draw Cross-Section Grid & Axes
        cv2.circle(panel, (cx, cy), int(15 * scale_plot), (30, 41, 59), 1)
        cv2.line(panel, (cx - 180, cy), (cx + 180, cy), (51, 65, 85), 1)
        cv2.line(panel, (cx, cy - 180), (cx, cy + 180), (51, 65, 85), 1)

        # 1. Naive Ellipse (Dotted Red)
        a_ell = (width_front_cm / 2.0) * scale_plot
        b_ell = (depth_right_cm / 2.0) * scale_plot
        cv2.ellipse(panel, (cx, cy), (int(a_ell), int(b_ell)), 0, 0, 360, (248, 113, 113), 1, cv2.LINE_AA)

        # 2. Reconstructed Lordosis Spline (Solid Green)
        c_nodes = recon_res.contour_points
        pts_contour = []
        for x_cm, y_cm in c_nodes:
            px = int(cx + x_cm * scale_plot)
            py = int(cy - y_cm * scale_plot)
            pts_contour.append([px, py])
        if len(pts_contour) > 2:
            cv2.polylines(panel, [np.array(pts_contour, dtype=np.int32)], True, (74, 222, 128), 2, cv2.LINE_AA)

        # Anatomical Markers
        cv2.circle(panel, (cx, int(cy + (depth_right_cm / 2.0 - recon_res.lordosis_depth_cm) * scale_plot)), 4, (250, 204, 21), -1)
        cv2.putText(panel, "Lumbar Spine", (cx + 8, int(cy + (depth_right_cm / 2.0 - recon_res.lordosis_depth_cm) * scale_plot)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (250, 204, 21), 1)

        # Right Half: Biometrics & ML Telemetry Card (X: 420 - 780)
        card_x1, card_y1 = 410, 48
        card_x2, card_y2 = 785, p_h - 15

        cv2.rectangle(panel, (card_x1, card_y1), (card_x2, card_y2), (15, 23, 42), -1)
        cv2.rectangle(panel, (card_x1, card_y1), (card_x2, card_y2), (56, 189, 248), 1)

        # Big Perimeter Readout
        cv2.putText(panel, f"{self.selected_site.value.upper()} PERIMETER", (card_x1 + 15, card_y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 1)
        cv2.putText(panel, f"{recon_res.perimeter_cm:.2f} cm", (card_x1 + 15, card_y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (74, 222, 128), 2)

        # Metrics Breakdown
        lines = [
            f"Coronal Width (X)  : {recon_res.coronal_width_cm:.2f} cm",
            f"Sagittal Depth (Y) : {recon_res.sagittal_depth_cm:.2f} cm",
            f"Aspect Ratio (W/D) : {recon_res.aspect_ratio:.2f}",
            f"Cross-Sect Area    : {recon_res.cross_sectional_area_cm2:.1f} cm^2",
            f"Lordosis Furrow    : {recon_res.lordosis_depth_cm:.2f} cm",
            f"Superellipse p*    : {recon_res.superellipse_p:.2f}",
            f"Calibration Status : {'VERIFIED' if self.pixels_per_cm > 5 else 'DEFAULT'}",
        ]

        for i, text in enumerate(lines):
            cv2.putText(panel, text, (card_x1 + 15, card_y1 + 105 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (203, 213, 225), 1)

        return panel

    # =========================================================================
    # Master HUD Overlay on Top of Combined Canvas
    # =========================================================================
    def _draw_master_hud(self, canvas: np.ndarray, fps: float) -> None:
        cw = canvas.shape[1]
        
        # Bottom Status Ribbon
        cv2.rectangle(canvas, (0, canvas.shape[0] - 28), (cw, canvas.shape[0]), (15, 23, 42), -1)
        cv2.line(canvas, (0, canvas.shape[0] - 28), (cw, canvas.shape[0] - 28), (51, 65, 85), 1)
        
        src_text = f"SOURCE: {self.camera_source}" if not self.is_synthetic_camera else "SOURCE: SYNTHETIC SIMULATOR"
        hud_str = (
            f"[FPS: {fps:.1f}]  |  {src_text}  |  VIEW: {self.view_mode.value}  |  "
            f"[V] Cycle Views  [P] Phone IP Camera  [S] Switch Site  [T] Toggle Simulator  [Q] Quit"
        )
        cv2.putText(canvas, hud_str, (16, canvas.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1)


def main():
    parser = argparse.ArgumentParser(description="Multi-View Calibration Dashboard & Biomechanical Workbench")
    parser.add_argument("--camera", default=0, help="Webcam index (0, 1) or Phone IP Camera URL (http://192.168.1.x:8080/video)")
    args = parser.parse_args()

    # Parse camera source as integer if numeric
    cam_src: Union[int, str] = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    app = CalibrationDashboardApp(camera_source=cam_src)
    app.run()


if __name__ == "__main__":
    main()
