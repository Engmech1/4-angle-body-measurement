"""
Sub-Pixel Edge Detection Module.

Extracts body silhouette boundaries with sub-pixel precision (< 0.1 pixel error)
along horizontal anatomical scanlines using 1D Derivative-of-Gaussian (DoG) filtering,
continuous parabolic peak interpolation, and zero-crossing of second spatial derivatives.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class EdgeSliceResult:
    """Sub-pixel boundary edge detection result for a single horizontal scanline."""
    y_pixel: int
    left_edge_x: float      # Sub-pixel coordinate of left body boundary
    right_edge_x: float     # Sub-pixel coordinate of right body boundary
    width_pixels: float     # Exact sub-pixel width (right_x - left_x)
    center_x: float         # Midline center of mass ((right_x + left_x) / 2.0)
    confidence: float       # SNR / gradient sharpness metric (0.0 to 1.0)
    is_valid: bool


class SubPixelEdgeDetector:
    """
    High-precision 1D sub-pixel edge detector for biometric cross-sectional slices.
    """

    def __init__(
        self,
        gaussian_sigma: float = 1.8,
        strip_half_height: int = 2,
        min_gradient_threshold: float = 10.0,
        psf_boundary_bias_px: float = 0.65,
    ):
        """
        Initializes the sub-pixel edge detector.

        Args:
            gaussian_sigma: Scale parameter for Derivative of Gaussian (DoG) filter.
            strip_half_height: Number of adjacent scanlines (y +/- k) to average for noise reduction.
            min_gradient_threshold: Minimum gradient magnitude to consider a valid body edge.
            psf_boundary_bias_px: Point spread function boundary offset compensation in pixels.
        """
        self.gaussian_sigma = gaussian_sigma
        self.strip_half_height = strip_half_height
        self.min_gradient_threshold = min_gradient_threshold
        self.psf_boundary_bias_px = float(psf_boundary_bias_px)

        # Precompute 1D Derivative of Gaussian (DoG) and 2nd Derivative (D2oG) kernels
        kernel_radius = int(np.ceil(3.5 * gaussian_sigma))
        x_vals = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float64)
        
        # 0th derivative (Gaussian)
        g = np.exp(-0.5 * (x_vals / gaussian_sigma) ** 2)
        g /= np.sum(g)
        self.g_kernel = g

        # 1st derivative of Gaussian: dG/dx = -x / sigma^2 * G(x)
        dog = -x_vals / (gaussian_sigma ** 2) * g
        # Normalize so response to unit step is 1.0
        dog /= (np.sum(np.abs(dog)) / 2.0 + 1e-12)
        self.dog_kernel = dog

        # 2nd derivative of Gaussian: d2G/dx2 = (x^2 - sigma^2) / sigma^4 * G(x)
        d2og = (x_vals ** 2 - gaussian_sigma ** 2) / (gaussian_sigma ** 4) * g
        self.d2og_kernel = d2og

    def extract_slice_edges(
        self,
        image: np.ndarray,
        y_slice: int,
        search_region_margin_ratio: float = 0.05,
        center_x_hint: Optional[float] = None,
        expected_half_width_px: Optional[float] = None,
    ) -> EdgeSliceResult:
        """
        Extracts sub-pixel left and right edges at scanline y_slice.

        Args:
            image: 2D Grayscale or 3D BGR image.
            y_slice: Target horizontal line Y.
            search_region_margin_ratio: Image margin to exclude from edge search.
            center_x_hint: Optional estimated center of torso X.
            expected_half_width_px: Optional expected torso half-width in pixels.

        Returns:
            EdgeSliceResult with sub-pixel edge coordinates and confidence.
        """
        if image is None or image.size == 0:
            return EdgeSliceResult(y_slice, 0.0, 0.0, 0.0, 0.0, 0.0, False)

        h, w = image.shape[:2]
        y_slice = int(np.clip(y_slice, 0, h - 1))

        # Convert to single-channel 8-bit or float grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Extract horizontal strip and average across Y-thickness to suppress random camera shot noise
        y_min = max(0, y_slice - self.strip_half_height)
        y_max = min(h, y_slice + self.strip_half_height + 1)
        strip = gray[y_min:y_max, :].astype(np.float64)
        profile_1d = np.mean(strip, axis=0)  # Shape (w,)

        # Compute 1st spatial derivative (gradient)
        grad_1d = np.convolve(profile_1d, self.dog_kernel, mode="same")
        grad_mag = np.abs(grad_1d)

        # Compute 2nd spatial derivative for zero-crossing check
        d2_1d = np.convolve(profile_1d, self.d2og_kernel, mode="same")

        # Define search windows for Left Edge and Right Edge
        margin = int(w * search_region_margin_ratio)
        mid_x = int(center_x_hint) if center_x_hint is not None else w // 2
        mid_x = int(np.clip(mid_x, margin + 20, w - margin - 20))

        if expected_half_width_px is not None:
            left_bound = max(margin, int(mid_x - expected_half_width_px * 1.8))
            right_bound = min(w - margin, int(mid_x + expected_half_width_px * 1.8))
        else:
            left_bound = margin
            right_bound = w - margin

        left_search = grad_mag[left_bound:mid_x]
        right_search = grad_mag[mid_x:right_bound]

        if len(left_search) < 5 or len(right_search) < 5:
            return EdgeSliceResult(y_slice, 0.0, 0.0, 0.0, 0.0, 0.0, False)

        # 1. Left Edge: Find strongest peak in left region
        left_peak_idx = left_bound + int(np.argmax(left_search))
        left_peak_val = grad_mag[left_peak_idx]

        # 2. Right Edge: Find strongest peak in right region
        right_peak_idx = mid_x + int(np.argmax(right_search))
        right_peak_val = grad_mag[right_peak_idx]

        # Refine Left Edge to Sub-Pixel accuracy using Parabolic Interpolation
        sub_left_x = self._subpixel_parabolic_peak(grad_mag, left_peak_idx, d2_1d)
        
        # Refine Right Edge to Sub-Pixel accuracy
        sub_right_x = self._subpixel_parabolic_peak(grad_mag, right_peak_idx, d2_1d)

        raw_width_px = sub_right_x - sub_left_x
        width_pixels = max(0.0, raw_width_px - self.psf_boundary_bias_px)
        center_x = (sub_left_x + sub_right_x) / 2.0

        # Confidence metric based on edge sharpness and signal-to-noise ratio
        noise_floor = float(np.median(grad_mag)) + 1e-6
        snr_left = left_peak_val / noise_floor
        snr_right = right_peak_val / noise_floor
        min_snr = min(snr_left, snr_right)
        # Measure edge transition width (FWHM) to reject motion-blurred / unsharp boundaries
        def _calc_fwhm(idx: int, val: float) -> int:
            half_val = val / 2.0
            l_idx = idx
            while l_idx > 0 and grad_mag[l_idx] > half_val:
                l_idx -= 1
            r_idx = idx
            while r_idx < len(grad_mag) - 1 and grad_mag[r_idx] > half_val:
                r_idx += 1
            return r_idx - l_idx

        fwhm_left = _calc_fwhm(left_peak_idx, left_peak_val)
        fwhm_right = _calc_fwhm(right_peak_idx, right_peak_val)
        max_fwhm = max(fwhm_left, fwhm_right)
        min_peak = min(left_peak_val, right_peak_val)

        is_valid = (
            width_pixels > 20.0
            and min_peak >= self.min_gradient_threshold
            and min_snr >= 2.0
            and max_fwhm <= 15.0
            and sub_left_x < sub_right_x
        )

        confidence = max(0.0, min(1.0, (min_snr - 1.0) / 9.0)) if is_valid else 0.0

        return EdgeSliceResult(
            y_pixel=y_slice,
            left_edge_x=float(sub_left_x),
            right_edge_x=float(sub_right_x),
            width_pixels=float(width_pixels),
            center_x=float(center_x),
            confidence=float(confidence),
            is_valid=is_valid,
        )

    def _subpixel_parabolic_peak(
        self,
        grad_mag: np.ndarray,
        peak_idx: int,
        d2_1d: Optional[np.ndarray] = None,
    ) -> float:
        """
        Fits a continuous quadratic parabola around peak_idx:
        x* = x0 + (y_{-1} - y_{+1}) / (2 * (y_{-1} - 2*y_0 + y_{+1}))
        """
        n = len(grad_mag)
        if peak_idx <= 0 or peak_idx >= n - 1:
            return float(peak_idx)

        y_prev = grad_mag[peak_idx - 1]
        y_curr = grad_mag[peak_idx]
        y_next = grad_mag[peak_idx + 1]

        denom = 2.0 * (y_prev - 2.0 * y_curr + y_next)
        if abs(denom) < 1e-12:
            return float(peak_idx)

        delta = (y_prev - y_next) / denom

        # Constrain delta to [-1.0, 1.0] interval
        delta = max(-1.0, min(1.0, delta))
        parabolic_x = float(peak_idx) + delta

        # Cross-validate with 2nd derivative zero-crossing if available
        if d2_1d is not None:
            # Check adjacent indices for zero crossing in d2_1d
            for i in [peak_idx - 1, peak_idx]:
                if 0 <= i < n - 1:
                    d2_a = d2_1d[i]
                    d2_b = d2_1d[i + 1]
                    if d2_a * d2_b <= 0 and abs(d2_b - d2_a) > 1e-12:
                        # Linear zero-crossing: x = i + (-d2_a) / (d2_b - d2_a)
                        zc_x = float(i) + (-d2_a) / (d2_b - d2_a)
                        # Blend parabolic and zero-crossing for maximum stability
                        if abs(zc_x - parabolic_x) < 0.75:
                            return 0.5 * (parabolic_x + zc_x)

        return parabolic_x
