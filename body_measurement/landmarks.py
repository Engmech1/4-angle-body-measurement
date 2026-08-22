"""
Anatomical Anchoring Module with Enterprise Error Handling & Occlusion Fallbacks.

Uses MediaPipe Pose (33 anatomical keypoints) to establish a normalized, invariant
Y-axis anatomical coordinate system. Ensures that measurement slices (Waist, Chest,
Hips, Thighs) are taken at the exact same anatomical height across weekly sessions.

Resilient to:
- Missing/occluded hip or shoulder landmarks
- Frame drops and corrupted input arrays
- Lateral view (90°/270°) joint overlap
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BodySite(str, Enum):
    """Supported standardized anthropometric measurement sites."""
    CHEST = "chest"
    WAIST = "waist"
    HIPS = "hips"
    THIGH = "thigh"
    CALF = "calf"


@dataclass
class AnatomicalAnchorResult:
    """Result of anatomical anchoring for a specific frame or view."""
    site: BodySite
    slice_y_pixel: int
    slice_y_normalized: float
    torso_height_pixels: float
    landmarks_detected: bool
    confidence: float
    keypoints_summary: Dict[str, Tuple[float, float]]
    fallback_used: bool = False
    status_note: Optional[str] = None


class AnatomicalAnchorEngine:
    """
    Computes invariant anatomical slice heights from 2D pose landmarks with robust fallbacks.
    """

    # ISAK & Biomechanical standard anatomical height ratios relative to torso span
    DEFAULT_SITE_RATIOS = {
        BodySite.CHEST: 0.35,   # ~4th intercostal space / mid-sternum
        BodySite.WAIST: 0.618,  # Golden ratio / narrowest natural waist between 10th rib & iliac crest
        BodySite.HIPS: 1.15,    # 15% below hip line (greater trochanter / maximum gluteal protrusion)
        BodySite.THIGH: 1.45,   # Mid-thigh level
        BodySite.CALF: 2.10,    # Maximum calf circumference
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._pose_detector = None
        self._last_valid_slice_norm: Optional[float] = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Safely initializes MediaPipe Pose detector."""
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self._pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=self.model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                logger.info("MediaPipe Pose detector initialized successfully.")
        except Exception as e:
            logger.warning(f"MediaPipe Pose detector unavailable ({e}). Using geometric prior fallback.")
            self._pose_detector = None

    def compute_anchor_slice(
        self,
        image: Optional[np.ndarray],
        site: BodySite = BodySite.WAIST,
        custom_ratio: Optional[float] = None,
    ) -> AnatomicalAnchorResult:
        """
        Extracts pose keypoints and computes the exact Y-pixel coordinate for the measurement slice.
        Guaranteed not to crash on corrupted images, occlusions, or missing landmarks.
        """
        # 1. Image Validation Guard
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            logger.warning("Empty frame provided for anatomical anchoring. Employing default framing fallback.")
            return self._create_default_fallback(site, 1920, 1080, "Input frame is None or empty.")

        h, w = image.shape[:2]
        ratio = custom_ratio if custom_ratio is not None else self.DEFAULT_SITE_RATIOS[site]

        keypoints: Dict[str, Tuple[float, float]] = {}
        detected = False
        confidence = 0.0

        # 2. Run MediaPipe Pose under safe try-except
        if self._pose_detector is not None:
            try:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

                results = self._pose_detector.process(rgb)
                if results and results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # Check individual joint visibility thresholds
                    vis_sh_l = float(getattr(lm[11], "visibility", 0.0))
                    vis_sh_r = float(getattr(lm[12], "visibility", 0.0))
                    vis_hip_l = float(getattr(lm[23], "visibility", 0.0))
                    vis_hip_r = float(getattr(lm[24], "visibility", 0.0))

                    # Shoulders: Handle unilateral occlusion in profile views
                    if vis_sh_l > 0.4 and vis_sh_r > 0.4:
                        y_shoulder = ((lm[11].y + lm[12].y) / 2.0) * h
                        keypoints["left_shoulder"] = (lm[11].x * w, lm[11].y * h)
                        keypoints["right_shoulder"] = (lm[12].x * w, lm[12].y * h)
                    elif vis_sh_l > 0.4:
                        y_shoulder = lm[11].y * h
                        keypoints["left_shoulder"] = (lm[11].x * w, lm[11].y * h)
                    elif vis_sh_r > 0.4:
                        y_shoulder = lm[12].y * h
                        keypoints["right_shoulder"] = (lm[12].x * w, lm[12].y * h)
                    else:
                        y_shoulder = None

                    # Hips: Handle unilateral hip occlusion
                    if vis_hip_l > 0.4 and vis_hip_r > 0.4:
                        y_hip = ((lm[23].y + lm[24].y) / 2.0) * h
                        keypoints["left_hip"] = (lm[23].x * w, lm[23].y * h)
                        keypoints["right_hip"] = (lm[24].x * w, lm[24].y * h)
                    elif vis_hip_l > 0.4:
                        y_hip = lm[23].y * h
                        keypoints["left_hip"] = (lm[23].x * w, lm[23].y * h)
                    elif vis_hip_r > 0.4:
                        y_hip = lm[24].y * h
                        keypoints["right_hip"] = (lm[24].x * w, lm[24].y * h)
                    else:
                        y_hip = None

                    if y_shoulder is not None and y_hip is not None:
                        torso_height = abs(y_hip - y_shoulder)
                        if torso_height >= (0.15 * h):  # Anatomically plausible torso span
                            slice_y_val = y_shoulder + ratio * torso_height
                            slice_y_pixel = int(np.clip(np.round(slice_y_val), 0, h - 1))
                            slice_y_norm = slice_y_pixel / float(h)
                            confidence = float(np.mean([vis_sh_l, vis_sh_r, vis_hip_l, vis_hip_r]))

                            self._last_valid_slice_norm = slice_y_norm

                            return AnatomicalAnchorResult(
                                site=site,
                                slice_y_pixel=slice_y_pixel,
                                slice_y_normalized=slice_y_norm,
                                torso_height_pixels=float(torso_height),
                                landmarks_detected=True,
                                confidence=confidence,
                                keypoints_summary=keypoints,
                                fallback_used=False,
                                status_note="Full anatomical landmarks extracted successfully.",
                            )
            except Exception as e:
                logger.warning(f"MediaPipe inference failed: {e}. Falling back gracefully.")

        # 3. Graceful Fallback to Historical Normalized Ratio or Geometric Prior
        return self._create_default_fallback(
            site, h, w, "Landmark tracking partially occluded or below confidence threshold."
        )

    def _create_default_fallback(
        self, site: BodySite, h: int, w: int, reason: str
    ) -> AnatomicalAnchorResult:
        """Creates a stabilized geometric anatomical prior fallback."""
        if self._last_valid_slice_norm is not None:
            norm_y = self._last_valid_slice_norm
            note = f"Used previous valid normalized slice ({norm_y:.3f}). Reason: {reason}"
        else:
            # Standard tripod standing framing prior:
            # Shoulders at ~0.25H, Hips at ~0.55H
            ratio = self.DEFAULT_SITE_RATIOS[site]
            norm_y = 0.25 + ratio * (0.55 - 0.25)
            note = f"Used standard biometric standing frame prior ({norm_y:.3f}). Reason: {reason}"

        slice_y_pixel = int(np.clip(np.round(norm_y * h), 0, h - 1))
        torso_h = 0.30 * h

        return AnatomicalAnchorResult(
            site=site,
            slice_y_pixel=slice_y_pixel,
            slice_y_normalized=norm_y,
            torso_height_pixels=torso_h,
            landmarks_detected=False,
            confidence=0.50,
            keypoints_summary={},
            fallback_used=True,
            status_note=note,
        )
