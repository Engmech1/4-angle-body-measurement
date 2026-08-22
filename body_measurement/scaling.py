"""
ArUco Metric Scaling & 3D Plane Pose Estimation Module.

Implements high-precision metric scaling (Pixels-per-Centimeter) with:
1. Sub-pixel ArUco fiducial corner refinement (cv2.cornerSubPix).
2. solvePnP 3D planar normal estimation and tilt angle evaluation.
3. Strict tilt rejection (rejects frames where marker normal deviates > 15 deg from optical axis).
4. Subject-plane perspective depth ratio correction: PPM_subject = PPM_wall * (Z_wall / Z_subject).
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from calibrate_camera import CameraCalibrator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Represents the metric calibration result from ArUco detection."""
    pixels_per_cm: float
    marker_id: int
    corners: np.ndarray             # Shape: (4, 2) sub-pixel corners
    reprojection_error: float
    is_valid: bool
    scale_confidence: float
    depth_correction_factor: float
    tilt_angle_deg: float = 0.0
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    error_message: Optional[str] = None


class ArucoMetricScaler:
    """
    High-precision ArUco metric scaler with 3D pose verification and tilt gating.
    """

    def __init__(
        self,
        marker_size_cm: float = 15.0,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        subpix_win_size: int = 5,
        fallback_pixels_per_cm: Optional[float] = None,
        camera_intrinsics: Optional[Dict[str, Any]] = None,
    ):
        self.marker_size_cm = float(marker_size_cm)
        self.dictionary_id = dictionary_id
        self.subpix_win_size = subpix_win_size
        self.fallback_pixels_per_cm = fallback_pixels_per_cm
        self.intrinsics = camera_intrinsics or CameraCalibrator.load_intrinsics()

        try:
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector_params.cornerRefinementWinSize = subpix_win_size
            self.detector_params.cornerRefinementMaxIterations = 40
            self.detector_params.cornerRefinementMinAccuracy = 0.005

            if hasattr(cv2.aruco, "ArucoDetector"):
                self._detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
            else:
                self._detector = None
        except Exception as e:
            logger.error(f"Failed to initialize ArUco detector: {e}")
            self._detector = None

    def detect_and_calibrate(
        self,
        image: Optional[np.ndarray],
        target_marker_id: Optional[int] = None,
        distance_camera_to_wall_cm: Optional[float] = None,
        distance_camera_to_subject_cm: Optional[float] = None,
        max_allowed_tilt_deg: float = 15.0,
    ) -> CalibrationResult:
        """
        Detects the ArUco marker, refines corners, estimates 3D tilt pose, and computes PPM.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return CalibrationResult(
                0.0, -1, np.empty((0, 2)), 999.0, False, 0.0, 1.0, error_message="Empty or invalid image."
            )

        # Apply lens undistortion if intrinsics are available
        if self.intrinsics:
            image = CameraCalibrator.undistort(image, self.intrinsics)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        # Detect markers
        try:
            if self._detector is not None:
                corners_list, ids, _ = self._detector.detectMarkers(gray)
            else:
                corners_list, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_params)
        except Exception as e:
            return CalibrationResult(
                0.0, -1, np.empty((0, 2)), 999.0, False, 0.0, 1.0, error_message=f"ArUco detection error: {e}"
            )

        if ids is None or len(ids) == 0:
            return CalibrationResult(
                0.0, -1, np.empty((0, 2)), 999.0, False, 0.0, 1.0, error_message="No ArUco marker detected."
            )

        # Select target marker
        flat_ids = ids.flatten()
        idx = 0
        if target_marker_id is not None:
            matches = np.where(flat_ids == target_marker_id)[0]
            if len(matches) == 0:
                return CalibrationResult(
                    0.0, -1, np.empty((0, 2)), 999.0, False, 0.0, 1.0,
                    error_message=f"Target marker ID {target_marker_id} not found.",
                )
            idx = matches[0]

        chosen_id = int(flat_ids[idx])
        raw_corners = corners_list[idx][0]  # Shape: (4, 2)

        # Sub-pixel corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        refined_corners = cv2.cornerSubPix(
            gray,
            raw_corners.astype(np.float32),
            (self.subpix_win_size, self.subpix_win_size),
            (-1, -1),
            criteria,
        )

        # 3D Object points for solvePnP
        s = self.marker_size_cm
        obj_pts = np.array([
            [-s / 2.0, s / 2.0, 0.0],
            [s / 2.0, s / 2.0, 0.0],
            [s / 2.0, -s / 2.0, 0.0],
            [-s / 2.0, -s / 2.0, 0.0],
        ], dtype=np.float64)

        # Camera matrix
        if self.intrinsics:
            cam_matrix = np.array(self.intrinsics["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.array(self.intrinsics["dist_coefficients"], dtype=np.float64)
        else:
            focal = 1400.0 * (w / 1920.0)
            cam_matrix = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)

        # Compute 3D pose and normal vector
        success, rvec, tvec = cv2.solvePnP(obj_pts, refined_corners, cam_matrix, dist_coeffs)
        tilt_deg = 0.0
        if success:
            R, _ = cv2.Rodrigues(rvec)
            normal = R @ np.array([0.0, 0.0, 1.0])
            # Angle between normal and camera Z axis [0, 0, 1]
            cos_tilt = np.clip(np.abs(normal[2]), -1.0, 1.0)
            tilt_deg = float(np.degrees(np.arccos(cos_tilt)))

            if tilt_deg > max_allowed_tilt_deg:
                return CalibrationResult(
                    pixels_per_cm=0.0,
                    marker_id=chosen_id,
                    corners=refined_corners,
                    reprojection_error=999.0,
                    is_valid=False,
                    scale_confidence=0.0,
                    depth_correction_factor=1.0,
                    tilt_angle_deg=tilt_deg,
                    rvec=rvec,
                    tvec=tvec,
                    error_message=f"ArUco plane tilt ({tilt_deg:.1f} deg) exceeds max allowed {max_allowed_tilt_deg:.1f} deg.",
                )

        # Compute edge lengths
        e0 = np.linalg.norm(refined_corners[1] - refined_corners[0])
        e1 = np.linalg.norm(refined_corners[2] - refined_corners[1])
        e2 = np.linalg.norm(refined_corners[3] - refined_corners[2])
        e3 = np.linalg.norm(refined_corners[0] - refined_corners[3])
        mean_edge_px = float((e0 + e1 + e2 + e3) / 4.0)

        # Dual-diagonal cross check
        d1 = np.linalg.norm(refined_corners[2] - refined_corners[0])
        d2 = np.linalg.norm(refined_corners[3] - refined_corners[1])
        diag_ratio = abs(d1 - d2) / max(d1, d2)

        raw_ppm = mean_edge_px / self.marker_size_cm

        # Depth ratio correction
        depth_correction = 1.0
        if distance_camera_to_wall_cm and distance_camera_to_subject_cm:
            depth_correction = float(distance_camera_to_wall_cm / max(1.0, distance_camera_to_subject_cm))

        final_ppm = raw_ppm * depth_correction
        confidence = float(np.clip(1.0 - (diag_ratio * 4.0), 0.1, 1.0))

        return CalibrationResult(
            pixels_per_cm=final_ppm,
            marker_id=chosen_id,
            corners=refined_corners,
            reprojection_error=float(diag_ratio * mean_edge_px),
            is_valid=True,
            scale_confidence=confidence,
            depth_correction_factor=depth_correction,
            tilt_angle_deg=tilt_deg,
            rvec=rvec if success else None,
            tvec=tvec if success else None,
        )
