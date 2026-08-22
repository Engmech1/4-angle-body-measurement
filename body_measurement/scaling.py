"""
ArUco Metric Scaling Module with Enterprise Error Handling & Resilience.

Uses OpenCV ArUco marker detection with sub-pixel corner refinement to compute
an exact Pixels-per-Metric (PPM) ratio (pixels per cm) for camera calibration.
Includes occlusion handling, frame corruption checks, logging, and depth offset correction.
"""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Represents the metric calibration result from ArUco detection."""
    pixels_per_cm: float
    marker_id: int
    corners: np.ndarray  # Shape: (4, 2) sub-pixel corners
    reprojection_error: float
    is_valid: bool
    scale_confidence: float
    depth_correction_factor: float
    error_message: Optional[str] = None


class ArucoMetricScaler:
    """
    Computes precise metric scaling (Pixels-Per-Centimeter) from ArUco fiducial markers.

    Features:
    - Sub-pixel corner localization using cv2.cornerSubPix
    - Geometric perimeter-based and dual-diagonal cross-check for high accuracy
    - Distance-ratio scaling correction: PPM_subject = PPM_wall * (Z_wall / Z_subject)
    - Graceful fallback and error handling when marker is occluded or corrupted
    """

    def __init__(
        self,
        marker_size_cm: float = 15.0,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        subpix_win_size: int = 5,
        fallback_pixels_per_cm: Optional[float] = None,
    ):
        self.marker_size_cm = float(marker_size_cm)
        self.dictionary_id = dictionary_id
        self.subpix_win_size = subpix_win_size
        self.fallback_pixels_per_cm = fallback_pixels_per_cm
        self._last_valid_calibration: Optional[CalibrationResult] = None

        try:
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.detector_params = cv2.aruco.DetectorParameters()
            self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector_params.cornerRefinementWinSize = subpix_win_size
            self.detector_params.cornerRefinementMaxIterations = 40
            self.detector_params.cornerRefinementMinAccuracy = 0.01

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
    ) -> CalibrationResult:
        """
        Detects the ArUco marker in the input image and calculates the metric scale.
        Safely handles None, corrupted arrays, occlusions, and out-of-frame markers.
        """
        # Calculate depth factor if distances provided
        depth_factor = 1.0
        if (
            distance_camera_to_wall_cm is not None
            and distance_camera_to_subject_cm is not None
            and distance_camera_to_subject_cm > 0
        ):
            depth_factor = float(distance_camera_to_wall_cm / distance_camera_to_subject_cm)

        # 1. Input Validation & Frame Corruption Guard
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            logger.warning("Empty or corrupted image provided to ArucoMetricScaler.")
            return self._handle_detection_failure("Input image is None or empty.", depth_factor)

        try:
            # 2. Convert to Grayscale
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray = image
            else:
                return self._handle_detection_failure(f"Unsupported image shape: {image.shape}", depth_factor)

            # 3. Marker Detection
            if self._detector is not None:
                corners_list, ids, rejected = self._detector.detectMarkers(gray)
            else:
                corners_list, ids, rejected = cv2.aruco.detectMarkers(
                    gray, self.dictionary, parameters=self.detector_params
                )

            if ids is None or len(ids) == 0:
                logger.info("ArUco marker not detected in image (occlusion or out of frame).")
                return self._handle_detection_failure("Marker not detected in frame.", depth_factor)

            # 4. Find Target Marker
            selected_idx = 0
            if target_marker_id is not None:
                found = False
                for i, mid in enumerate(ids.flatten()):
                    if mid == target_marker_id:
                        selected_idx = i
                        found = True
                        break
                if not found:
                    return self._handle_detection_failure(
                        f"Target marker ID {target_marker_id} not found among detected IDs {ids.flatten()}.",
                        depth_factor,
                    )

            raw_corners = corners_list[selected_idx][0].astype(np.float32)  # Shape (4, 2)
            marker_id = int(ids.flatten()[selected_idx])

            # 5. Sub-Pixel Corner Refinement
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                0.001,
            )
            try:
                refined_corners = cv2.cornerSubPix(
                    gray,
                    raw_corners,
                    winSize=(self.subpix_win_size, self.subpix_win_size),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )
            except Exception as e:
                logger.warning(f"Sub-pixel corner refinement failed: {e}. Using raw corners.")
                refined_corners = raw_corners

            # 6. Geometric Quality & Orthogonality Cross-Check
            edge_lengths = []
            for i in range(4):
                p1 = refined_corners[i]
                p2 = refined_corners[(i + 1) % 4]
                edge_len = float(np.linalg.norm(p2 - p1))
                edge_lengths.append(edge_len)

            mean_edge_pixels = float(np.mean(edge_lengths))
            edge_std = float(np.std(edge_lengths))

            d1 = float(np.linalg.norm(refined_corners[2] - refined_corners[0]))
            d2 = float(np.linalg.norm(refined_corners[3] - refined_corners[1]))
            diag_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0

            scale_confidence = max(
                0.0, min(1.0, 1.0 - (edge_std / (mean_edge_pixels + 1e-6)) * 2.0)
            ) * (diag_ratio ** 2)

            raw_ppm = mean_edge_pixels / self.marker_size_cm
            effective_ppm = raw_ppm * depth_factor

            result = CalibrationResult(
                pixels_per_cm=effective_ppm,
                marker_id=marker_id,
                corners=refined_corners,
                reprojection_error=edge_std,
                is_valid=bool(effective_ppm > 0.1 and scale_confidence > 0.4),
                scale_confidence=scale_confidence,
                depth_correction_factor=depth_factor,
                error_message=None,
            )

            if result.is_valid:
                self._last_valid_calibration = result

            return result

        except Exception as ex:
            logger.error(f"Unexpected error during ArUco calibration: {ex}", exc_info=True)
            return self._handle_detection_failure(f"Internal calibration exception: {ex}", depth_factor)

    def _handle_detection_failure(self, reason: str, depth_factor: float = 1.0) -> CalibrationResult:
        """Provides graceful fallback if previous valid calibration exists."""
        if self._last_valid_calibration is not None:
            logger.info("Using previous valid calibration as fallback.")
            return CalibrationResult(
                pixels_per_cm=self._last_valid_calibration.pixels_per_cm,
                marker_id=self._last_valid_calibration.marker_id,
                corners=self._last_valid_calibration.corners,
                reprojection_error=self._last_valid_calibration.reprojection_error,
                is_valid=True,
                scale_confidence=self._last_valid_calibration.scale_confidence * 0.9,
                depth_correction_factor=depth_factor,
                error_message=f"Fallback used: {reason}",
            )
        elif self.fallback_pixels_per_cm is not None and self.fallback_pixels_per_cm > 0:
            logger.info("Using default fallback PPM value.")
            return CalibrationResult(
                pixels_per_cm=self.fallback_pixels_per_cm * depth_factor,
                marker_id=-1,
                corners=np.empty((0, 2)),
                reprojection_error=0.0,
                is_valid=True,
                scale_confidence=0.5,
                depth_correction_factor=depth_factor,
                error_message=f"Default fallback used: {reason}",
            )
        else:
            return CalibrationResult(
                pixels_per_cm=0.0,
                marker_id=-1,
                corners=np.empty((0, 2)),
                reprojection_error=0.0,
                is_valid=False,
                scale_confidence=0.0,
                depth_correction_factor=depth_factor,
                error_message=reason,
            )
