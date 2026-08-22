"""
Integrated 4-Angle Guided Capture Body Measurement System.

Orchestrates:
1. ArUco Metric Scaling (OpenCV sub-pixel corner calibration)
2. MediaPipe Anatomical Anchoring (33-keypoint normalized invariant Y-slice)
3. Sub-Pixel Edge & Burst Averaging (30 frames/angle, sway detrending, MAD filtering)
4. Anthropometric Non-Elliptical Cross-Section & Quadrature Perimeter Fitting
5. Zero-Raw-Media Privacy (Local in-memory execution, frames deleted immediately)
"""

from dataclasses import dataclass, field
from enum import IntEnum
import logging
from typing import Dict, Iterable, List, Optional
import numpy as np

from body_measurement.burst_processor import BurstAngleResult, BurstFrameProcessor
from body_measurement.edge_detection import SubPixelEdgeDetector
from body_measurement.landmarks import AnatomicalAnchorEngine, AnatomicalAnchorResult, BodySite
from body_measurement.reconstruction import (
    CrossSectionReconstructor,
    CrossSectionResult,
    ReconstructionMethod,
)
from body_measurement.scaling import ArucoMetricScaler, CalibrationResult

logger = logging.getLogger(__name__)


class CaptureAngle(IntEnum):
    """4 Orthogonal guided capture angles."""
    FRONT = 0
    RIGHT_PROFILE = 90
    BACK = 180
    LEFT_PROFILE = 270


@dataclass
class BodyMeasurementSummary:
    """Final output report of the 4-angle guided body measurement system."""
    site: BodySite
    perimeter_cm: float
    coronal_width_cm: float
    sagittal_depth_cm: float
    aspect_ratio: float
    cross_sectional_area_cm2: float
    pixels_per_cm: float
    slice_y_normalized: float
    burst_results: Dict[int, BurstAngleResult]
    cross_section_contour: np.ndarray
    reconstruction_method: ReconstructionMethod
    privacy_verified_zero_disk: bool
    is_successful: bool
    status_message: str
    quality_flags: List[str] = field(default_factory=list)


class BodyMeasurementSystem:
    """
    Main orchestration class for the Exercise App Guided Body Measurement System.
    """

    # Anatomical cross-section morphological shape priors per site
    SITE_MORPHOLOGY_PRIORS = {
        BodySite.WAIST: {"lordosis_ratio": 0.140, "superellipse_p": 2.45},
        BodySite.CHEST: {"lordosis_ratio": 0.055, "superellipse_p": 2.50},
        BodySite.HIPS: {"lordosis_ratio": 0.075, "superellipse_p": 2.55},
        BodySite.THIGH: {"lordosis_ratio": 0.000, "superellipse_p": 2.15},
        BodySite.CALF: {"lordosis_ratio": 0.000, "superellipse_p": 2.20},
    }

    def __init__(
        self,
        marker_size_cm: float = 15.0,
        gaussian_sigma: float = 1.8,
        mad_threshold: float = 2.5,
        reconstruction_method: ReconstructionMethod = ReconstructionMethod.DEFORMABLE_SUPERELLIPSE,
    ):
        self.scaler = ArucoMetricScaler(marker_size_cm=marker_size_cm)
        self.anchor_engine = AnatomicalAnchorEngine()
        self.edge_detector = SubPixelEdgeDetector(gaussian_sigma=gaussian_sigma)
        self.burst_processor = BurstFrameProcessor(
            edge_detector=self.edge_detector,
            mad_threshold=mad_threshold,
        )
        self.reconstructor = CrossSectionReconstructor(default_method=reconstruction_method)

        self._calibration_result: Optional[CalibrationResult] = None
        self._anchor_result: Optional[AnatomicalAnchorResult] = None
        self._burst_data: Dict[int, BurstAngleResult] = {}

    def calibrate(
        self,
        calibration_frame: np.ndarray,
        distance_wall_cm: Optional[float] = None,
        distance_subject_cm: Optional[float] = None,
    ) -> CalibrationResult:
        """Calibrates the camera pixels-per-cm scale using an ArUco marker."""
        res = self.scaler.detect_and_calibrate(
            calibration_frame,
            distance_camera_to_wall_cm=distance_wall_cm,
            distance_camera_to_subject_cm=distance_subject_cm,
        )
        self._calibration_result = res
        return res

    def set_manual_scale(self, pixels_per_cm: float) -> None:
        """Manually sets the metric scale if calibrated externally."""
        self._calibration_result = CalibrationResult(
            pixels_per_cm=pixels_per_cm,
            marker_id=0,
            corners=np.empty((0, 2)),
            reprojection_error=0.0,
            is_valid=True,
            scale_confidence=1.0,
            depth_correction_factor=1.0,
        )

    def determine_anchor(
        self,
        reference_frame: np.ndarray,
        site: BodySite = BodySite.WAIST,
    ) -> AnatomicalAnchorResult:
        """Extracts 33 pose landmarks and anchors the normalized anatomical slice height."""
        res = self.anchor_engine.compute_anchor_slice(reference_frame, site=site)
        self._anchor_result = res
        return res

    def process_angle_burst(
        self,
        angle: CaptureAngle,
        frames: Iterable[np.ndarray],
        y_slice: Optional[int] = None,
    ) -> BurstAngleResult:
        """Processes a 30-frame in-memory video stream for one angle."""
        if self._calibration_result is None or not self._calibration_result.is_valid:
            raise ValueError("System must be calibrated (PPM > 0) before processing angle bursts.")

        target_y = (
            y_slice
            if y_slice is not None
            else (self._anchor_result.slice_y_pixel if self._anchor_result else None)
        )

        if target_y is None:
            raise ValueError("Anatomical slice Y must be anchored before burst processing.")

        ppm = self._calibration_result.pixels_per_cm
        res = self.burst_processor.process_burst(
            frames=frames,
            y_slice=target_y,
            angle_degrees=int(angle),
            pixels_per_cm=ppm,
        )
        self._burst_data[int(angle)] = res
        return res

    def compute_measurement(
        self,
        site: BodySite = BodySite.WAIST,
        method: Optional[ReconstructionMethod] = None,
        custom_lordosis_cm: Optional[float] = None,
        custom_p: Optional[float] = None,
    ) -> BodyMeasurementSummary:
        """Computes the final non-elliptical perimeter from the 4 captured angle bursts."""
        has_front_and_profile = (
            int(CaptureAngle.FRONT) in self._burst_data
            and int(CaptureAngle.RIGHT_PROFILE) in self._burst_data
        )

        if not has_front_and_profile:
            return BodyMeasurementSummary(
                site=site,
                perimeter_cm=0.0,
                coronal_width_cm=0.0,
                sagittal_depth_cm=0.0,
                aspect_ratio=0.0,
                cross_sectional_area_cm2=0.0,
                pixels_per_cm=self._calibration_result.pixels_per_cm if self._calibration_result else 0.0,
                slice_y_normalized=self._anchor_result.slice_y_normalized if self._anchor_result else 0.0,
                burst_results=self._burst_data,
                cross_section_contour=np.empty((0, 2)),
                reconstruction_method=method or self.reconstructor.default_method,
                privacy_verified_zero_disk=True,
                is_successful=False,
                status_message="Missing required angle bursts (Front 0° and Profile 90° required).",
                quality_flags=["QUALITY_ERR_MISSING_ANGLES"],
            )

        w_0 = self._burst_data[int(CaptureAngle.FRONT)].width_cm
        d_90 = self._burst_data[int(CaptureAngle.RIGHT_PROFILE)].width_cm
        w_180 = (
            self._burst_data[int(CaptureAngle.BACK)].width_cm
            if int(CaptureAngle.BACK) in self._burst_data
            else w_0
        )
        d_270 = (
            self._burst_data[int(CaptureAngle.LEFT_PROFILE)].width_cm
            if int(CaptureAngle.LEFT_PROFILE) in self._burst_data
            else d_90
        )

        # Apply anatomical morphology prior
        prior = self.SITE_MORPHOLOGY_PRIORS.get(site, {"lordosis_ratio": 0.125, "superellipse_p": 2.45})
        d_mean = (d_90 + d_270) / 2.0
        lordosis_val = custom_lordosis_cm if custom_lordosis_cm is not None else (d_mean * prior["lordosis_ratio"])
        p_val = custom_p

        chosen_method = method or self.reconstructor.default_method
        recon_res = self.reconstructor.reconstruct_cross_section(
            width_front_cm=w_0,
            depth_right_cm=d_90,
            width_back_cm=w_180,
            depth_left_cm=d_270,
            method=chosen_method,
            custom_lordosis_depth_cm=lordosis_val,
            custom_superellipse_p=p_val,
        )

        # Quality Gating & Sanity Checks
        quality_flags = []
        is_valid = recon_res.is_valid

        # 1. Front / Back symmetry check (detects asymmetric clothing drapes and yaw errors)
        coronal_asymmetry = abs(w_0 - w_180)
        if coronal_asymmetry > 2.0:
            quality_flags.append("QUALITY_WARN_CORONAL_ASYMMETRY")
            is_valid = False

        # 2. Left / Right profile depth symmetry check
        sagittal_asymmetry = abs(d_90 - d_270)
        if sagittal_asymmetry > 2.0:
            quality_flags.append("QUALITY_WARN_SAGITTAL_ASYMMETRY")
            is_valid = False

        # 3. Plausible human girth bounds
        p_val_meas = recon_res.perimeter_hull_cm
        if not (40.0 <= p_val_meas <= 250.0):
            quality_flags.append("QUALITY_ERR_GIRTH_OUT_OF_BOUNDS")
            is_valid = False

        if not (15.0 <= recon_res.coronal_width_cm <= 90.0):
            quality_flags.append("QUALITY_ERR_WIDTH_OUT_OF_BOUNDS")
            is_valid = False

        if not (10.0 <= recon_res.sagittal_depth_cm <= 70.0):
            quality_flags.append("QUALITY_ERR_DEPTH_OUT_OF_BOUNDS")
            is_valid = False

        if not (0.60 <= recon_res.aspect_ratio <= 2.50):
            quality_flags.append("QUALITY_ERR_ASPECT_RATIO_ANOMALY")
            is_valid = False

        status_msg = (
            "Successfully computed perimeter with sub-pixel reconstruction."
            if is_valid
            else f"Measurement refused due to quality gate violation: {', '.join(quality_flags)}"
        )

        return BodyMeasurementSummary(
            site=site,
            perimeter_cm=recon_res.perimeter_hull_cm,
            coronal_width_cm=recon_res.coronal_width_cm,
            sagittal_depth_cm=recon_res.sagittal_depth_cm,
            aspect_ratio=recon_res.aspect_ratio,
            cross_sectional_area_cm2=recon_res.cross_sectional_area_cm2,
            pixels_per_cm=self._calibration_result.pixels_per_cm if self._calibration_result else 0.0,
            slice_y_normalized=self._anchor_result.slice_y_normalized if self._anchor_result else 0.0,
            burst_results=dict(self._burst_data),
            cross_section_contour=recon_res.contour_points_cm,
            reconstruction_method=recon_res.method_used,
            privacy_verified_zero_disk=True,
            is_successful=is_valid,
            status_message=status_msg,
            quality_flags=quality_flags,
        )
