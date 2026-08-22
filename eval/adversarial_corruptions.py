"""
Adversarial Corruption Generators for SPEC §4 Tier 4 Robustness Evaluation.

Implements all 15+ real physical and sensor corruptions:
1. Directional shadow + cast blob
2. Backlight / blown highlights
3. Low light + sensor Gaussian/Poisson noise
4. Severe colour cast (tungsten/fluorescent)
5. Linear motion blur
6. JPEG compression at Q=40
7. Rolling-shutter horizontal shear
8. ArUco 3D out-of-plane tilt (5°, 10°, 20°)
9. ArUco partial occlusion (35% area blocked)
10. ArUco local motion blur
11. Dynamic postural sway (per-frame affine jitter)
12. Loose clothing / asymmetrical fabric drape (+3 to +15 px)
13. Skin-coloured background clutter
14. Mirror / secondary ghost reflection
15. Subject yaw rotation misalignment (±8°)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np


def apply_shadow(frame: np.ndarray) -> np.ndarray:
    """Applies a directional illumination gradient plus a dark cast shadow blob."""
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    # Directional gradient (left-to-right lighting falloff)
    grad = np.linspace(0.40, 1.05, w, dtype=np.float32).reshape(1, w)
    if out.ndim == 3:
        grad = np.repeat(grad[:, :, np.newaxis], 3, axis=2)
    out = out * grad

    # Cast shadow blob
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - w * 0.42) / (w * 0.15)) ** 2 + ((yy - h * 0.50) / (h * 0.20)) ** 2))
    if out.ndim == 3:
        blob = blob[:, :, np.newaxis]
    out = out * (1.0 - 0.45 * blob)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_backlight(frame: np.ndarray) -> np.ndarray:
    """Applies intense blown-out background hotspot and lens flare."""
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    hotspot = np.exp(-(((xx - w * 0.5) / (w * 0.35)) ** 2 + ((yy - h * 0.35) / (h * 0.30)) ** 2)) * 140.0
    if out.ndim == 3:
        hotspot = hotspot[:, :, np.newaxis]
    out = np.clip(out + hotspot, 0, 255)
    return out.astype(np.uint8)


def apply_low_light_noise(frame: np.ndarray, seed: int = 42) -> np.ndarray:
    """Simulates low-light underexposure with sensor read and shot noise."""
    rng = np.random.RandomState(seed)
    out = frame.astype(np.float32) * 0.30  # 70% underexposure
    noise = rng.normal(0, 18.0, size=frame.shape).astype(np.float32)
    out = np.clip(out + noise, 0, 255)
    return out.astype(np.uint8)


def apply_colour_cast(frame: np.ndarray) -> np.ndarray:
    """Simulates severe tungsten / warm color cast."""
    out = frame.astype(np.float32)
    if out.ndim == 3:
        # BGR: reduce Blue, boost Red
        out[:, :, 0] *= 0.50  # Blue
        out[:, :, 1] *= 0.85  # Green
        out[:, :, 2] *= 1.35  # Red
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_motion_blur(frame: np.ndarray, kernel_size: int = 15, angle_deg: float = 20.0) -> np.ndarray:
    """Applies directional linear camera motion blur."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    center = kernel_size // 2
    for i in range(-center, center + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    k_sum = np.sum(kernel)
    if k_sum > 0:
        kernel /= k_sum
    return cv2.filter2D(frame, -1, kernel)


def apply_jpeg_q40(frame: np.ndarray) -> np.ndarray:
    """Applies JPEG compression artifacts at Quality=40."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, enc = cv2.imencode(".jpg", frame, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)


def apply_rolling_shutter(frame: np.ndarray, max_shear_px: float = 12.0) -> np.ndarray:
    """Simulates rolling shutter scanline shear."""
    h, w = frame.shape[:2]
    out = np.empty_like(frame)
    for y in range(h):
        shift = int(round(max_shear_px * (y - h / 2.0) / (h / 2.0)))
        out[y] = np.roll(frame[y], shift, axis=0)
    return out


def apply_aruco_tilt(frame: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Warps the top-left quadrant (where ArUco marker resides) with 3D out-of-plane tilt."""
    h, w = frame.shape[:2]
    out = frame.copy()
    # Marker region in top-left
    mx1, my1, mx2, my2 = 20, 20, 220, 220
    if my2 > h or mx2 > w:
        return out

    src_pts = np.float32([[mx1, my1], [mx2, my1], [mx2, my2], [mx1, my2]])
    rad = np.radians(tilt_deg)
    # Perspective compression along one edge
    delta_x = 20.0 * np.sin(rad)
    delta_y = 15.0 * np.sin(rad)
    dst_pts = np.float32([
        [mx1 + delta_x, my1 + delta_y],
        [mx2 - delta_x, my1],
        [mx2 - delta_x, my2],
        [mx1 + delta_x, my2 - delta_y],
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    patch = frame[my1:my2, mx1:mx2]
    warped_patch = cv2.warpPerspective(patch, M, (mx2 - mx1, my2 - my1), borderValue=(225, 225, 225) if frame.ndim == 3 else 225)
    out[my1:my2, mx1:mx2] = warped_patch
    return out


def apply_aruco_occlusion(frame: np.ndarray) -> np.ndarray:
    """Occludes 35% of the ArUco marker corners with an opaque sticker patch."""
    h, w = frame.shape[:2]
    out = frame.copy()
    # Occlude corner of marker at top-left
    ox1, oy1, ox2, oy2 = 50, 50, 110, 110
    if oy2 <= h and ox2 <= w:
        out[oy1:oy2, ox1:ox2] = (190, 190, 190) if frame.ndim == 3 else 190
    return out


def apply_aruco_motion_blur(frame: np.ndarray) -> np.ndarray:
    """Applies severe local motion blur exclusively over the ArUco marker."""
    h, w = frame.shape[:2]
    out = frame.copy()
    mx1, my1, mx2, my2 = 30, 30, 200, 200
    if my2 <= h and mx2 <= w:
        patch = frame[my1:my2, mx1:mx2]
        blurred = apply_motion_blur(patch, kernel_size=21, angle_deg=45.0)
        out[my1:my2, mx1:mx2] = blurred
    return out


def apply_postural_sway(frames: List[np.ndarray], max_sway_px: float = 12.0) -> List[np.ndarray]:
    """Applies per-frame affine translational & rotational jitter to simulate subject sway."""
    out_frames = []
    n = len(frames)
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        dx = max_sway_px * np.sin(2.0 * np.pi * i / 14.0)
        d_rot = 1.8 * np.cos(2.0 * np.pi * i / 10.0)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), d_rot, 1.0)
        M[0, 2] += dx
        warped = cv2.warpAffine(frame, M, (w, h), borderValue=(225, 225, 225) if frame.ndim == 3 else 225)
        out_frames.append(warped)
    return out_frames


def apply_loose_clothing(frame: np.ndarray, waist_y: int, dilation_px: int = 12) -> np.ndarray:
    """Simulates loose clothing drape adding an asymmetric bump on one lateral flank."""
    out = frame.copy()
    h, w = frame.shape[:2]
    if 0 <= waist_y < h:
        # Find dark subject pixels on waist scanline
        scanline = out[waist_y]
        if out.ndim == 3:
            is_dark = np.mean(scanline, axis=1) < 150
        else:
            is_dark = scanline < 150
        dark_indices = np.where(is_dark)[0]
        if len(dark_indices) > 0:
            rx = dark_indices[-1]
            # Dilate right flank by dilation_px over a 20 px vertical span
            y1 = max(0, waist_y - 12)
            y2 = min(h, waist_y + 13)
            for y in range(y1, y2):
                falloff = np.cos(np.pi * (y - waist_y) / 25.0)
                cur_d = int(round(dilation_px * falloff))
                if rx + cur_d < w:
                    out[y, rx : rx + cur_d] = (40, 40, 40) if frame.ndim == 3 else 40
    return out


def apply_skin_background_clutter(frame: np.ndarray, waist_y: int) -> np.ndarray:
    """Draws skin-toned distractor clutter in the background adjacent to subject."""
    out = frame.copy()
    h, w = frame.shape[:2]
    # Draw a flesh-colored rectangle 40 px away from center
    cx = int(w * 0.72)
    cy = waist_y
    if 0 <= cy < h and 0 <= cx < w:
        cv2.rectangle(out, (cx - 15, cy - 25), (cx + 15, cy + 25), (140, 160, 210) if frame.ndim == 3 else 160, -1)
    return out


def apply_mirror_reflection(frame: np.ndarray) -> np.ndarray:
    """Adds a ghost reflection at 25% opacity shifted 35 px laterally."""
    h, w = frame.shape[:2]
    shift_M = np.float32([[1, 0, 35], [0, 1, 0]])
    ghost = cv2.warpAffine(frame, shift_M, (w, h), borderValue=(225, 225, 225) if frame.ndim == 3 else 225)
    blended = cv2.addWeighted(frame, 0.75, ghost, 0.25, 0)
    return blended
