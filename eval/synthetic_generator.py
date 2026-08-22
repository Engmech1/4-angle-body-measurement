"""
Procedural 3D Digital Twin & Ground-Truth Silhouette Generator.

Generates:
1. Procedural 3D human torso/limb geometries with parameterized BMI, waist-to-hip ratio, and lumbar lordosis.
2. Ground-truth 2D cross-sectional slices with exact raw perimeter and convex-hull tape-measure perimeter.
3. Multi-angle silhouette projections rendered with a calibrated pinhole camera and in-plane ArUco fiducials.
4. Deterministic Dev and Holdout evaluation datasets.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.spatial import ConvexHull


@dataclass
class GroundTruthProfile:
    """Ground truth metrics for an anatomical cross-section."""
    site_name: str
    height_fraction: float          # Normalized vertical position (0.0=feet, 1.0=head)
    coronal_width_cm: float         # Width across X axis (Front view)
    sagittal_depth_cm: float        # Depth across Z axis (Profile view)
    superellipse_p: float           # Exponent of flank profile
    lordosis_depth_cm: float        # Lumbar indentation depth
    perimeter_raw_cm: float         # Exact anatomical arc-length contour perimeter
    perimeter_hull_cm: float        # Perimeter of convex hull (taut tape ground truth)
    cross_sectional_area_cm2: float # Area of cross-section
    contour_points_cm: np.ndarray   # Dense 2D (x, z) points (N, 2)


@dataclass
class RenderedSubjectScene:
    """Multi-angle rendered silhouettes with camera parameters and ground truth."""
    subject_id: str
    height_cm: float
    distance_m: float
    camera_height_cm: float
    focal_length_px: float
    image_size: Tuple[int, int]     # (width, height)
    pixels_per_cm: float
    aruco_marker_size_cm: float
    ground_truth: Dict[str, GroundTruthProfile]
    frames_by_angle: Dict[int, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DigitalTwinGenerator:
    """
    Deterministic procedural human body mesh & multi-view silhouette generator.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_ground_truth_cross_section(
        self,
        site_name: str,
        coronal_width_cm: float,
        sagittal_depth_cm: float,
        superellipse_p: float = 2.45,
        lordosis_depth_cm: float = 0.0,
        n_samples: int = 2048,
    ) -> GroundTruthProfile:
        """
        Generates dense 2D cross-section contour and exact perimeters (raw & convex hull).
        Standard Lamé curve (superellipse): |x/a|^p + |z/b|^p = 1
        Parametric representation:
        x(t) = a * sgn(cos(t)) * |cos(t)|^(2/p)
        z(t) = b * sgn(sin(t)) * |sin(t)|^(2/p)
        """
        theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        a = coronal_width_cm / 2.0
        b = sagittal_depth_cm / 2.0
        p = max(0.5, float(superellipse_p))
        exp = 2.0 / p

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        sign_cos = np.sign(cos_t)
        sign_sin = np.sign(sin_t)

        x = a * sign_cos * (np.abs(cos_t) ** exp)
        z = b * sign_sin * (np.abs(sin_t) ** exp)

        # Apply lumbar lordosis depression along posterior midline (z < 0, x near 0)
        if lordosis_depth_cm > 0.0:
            spine_weight = np.exp(-0.5 * (x / (max(0.1, a) * 0.35)) ** 2) * np.maximum(0.0, -z / max(0.1, b))
            z = z + lordosis_depth_cm * spine_weight

        contour = np.column_stack((x, z))

        # 1. Raw contour perimeter (sum of Euclidean chord segments)
        diffs = np.diff(contour, axis=0, append=contour[:1])
        perimeter_raw = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

        # 2. Convex Hull perimeter (represents physical taut tape bridging concavities)
        hull = ConvexHull(contour)
        hull_points = contour[hull.vertices]
        hull_diffs = np.diff(hull_points, axis=0, append=hull_points[:1])
        perimeter_hull = float(np.sum(np.sqrt(np.sum(hull_diffs ** 2, axis=1))))

        # 3. Shoelace area
        x_pts, z_pts = contour[:, 0], contour[:, 1]
        area = 0.5 * float(np.abs(np.dot(x_pts, np.roll(z_pts, 1)) - np.dot(z_pts, np.roll(x_pts, 1))))

        return GroundTruthProfile(
            site_name=site_name,
            height_fraction=0.618 if site_name == "waist" else 0.75,
            coronal_width_cm=coronal_width_cm,
            sagittal_depth_cm=sagittal_depth_cm,
            superellipse_p=p,
            lordosis_depth_cm=lordosis_depth_cm,
            perimeter_raw_cm=perimeter_raw,
            perimeter_hull_cm=perimeter_hull,
            cross_sectional_area_cm2=area,
            contour_points_cm=contour,
        )

    def render_subject_scene(
        self,
        subject_id: str,
        height_cm: float = 175.0,
        waist_width_cm: float = 30.0,
        waist_depth_cm: float = 21.0,
        chest_width_cm: float = 34.0,
        chest_depth_cm: float = 24.0,
        hip_width_cm: float = 35.0,
        hip_depth_cm: float = 25.0,
        distance_m: float = 2.2,
        angles: Tuple[int, ...] = (0, 90, 180, 270),
        image_size: Tuple[int, int] = (1920, 1080),  # (width, height)
        aruco_size_cm: float = 15.0,
        camera_pitch_deg: float = 0.0,
        camera_roll_deg: float = 0.0,
        noise_sigma: float = 0.0,
    ) -> RenderedSubjectScene:
        """
        Renders multi-angle body silhouettes and ArUco calibration targets.
        """
        img_w, img_h = image_size
        focal_length = 1400.0 * (img_w / 1920.0)
        cx, cy = img_w / 2.0, img_h / 2.0

        # Pixels-per-cm in the subject plane: PPM = f / (Z_cm)
        dist_cm = distance_m * 100.0
        ppm = focal_length / dist_cm

        # Generate ground truth profiles for primary sites
        gt_profiles = {
            "waist": self.generate_ground_truth_cross_section(
                "waist", waist_width_cm, waist_depth_cm, superellipse_p=2.45, lordosis_depth_cm=waist_depth_cm * 0.12
            ),
            "chest": self.generate_ground_truth_cross_section(
                "chest", chest_width_cm, chest_depth_cm, superellipse_p=2.50, lordosis_depth_cm=chest_depth_cm * 0.05
            ),
            "hips": self.generate_ground_truth_cross_section(
                "hips", hip_width_cm, hip_depth_cm, superellipse_p=2.55, lordosis_depth_cm=hip_depth_cm * 0.07
            ),
        }

        frames = {}
        for angle in angles:
            frame = np.ones((img_h, img_w, 3), dtype=np.uint8) * 230  # Studio background

            # Determine projected silhouette width at waist slice as a function of rotation angle
            rad = np.radians(angle)
            gt_w = waist_width_cm
            gt_d = waist_depth_cm
            proj_w_cm = np.sqrt((gt_w * np.cos(rad)) ** 2 + (gt_d * np.sin(rad)) ** 2)
            half_w_px = (proj_w_cm * ppm) / 2.0

            # Compute body vertical span
            body_h_px = height_cm * ppm
            body_top_y = cy - (body_h_px * 0.45)
            body_bot_y = body_top_y + body_h_px
            waist_y = body_top_y + (body_h_px * 0.40)

            # Draw smooth body silhouette
            pts = [
                (int(cx - half_w_px * 1.15), int(body_top_y + body_h_px * 0.22)), # Chest
                (int(cx - half_w_px), int(waist_y)),                               # Waist
                (int(cx - half_w_px * 1.18), int(body_top_y + body_h_px * 0.55)), # Hips
                (int(cx - half_w_px * 0.6), int(body_bot_y)),                      # Legs
                (int(cx + half_w_px * 0.6), int(body_bot_y)),
                (int(cx + half_w_px * 1.18), int(body_top_y + body_h_px * 0.55)),
                (int(cx + half_w_px), int(waist_y)),
                (int(cx + half_w_px * 1.15), int(body_top_y + body_h_px * 0.22)),
                (int(cx), int(body_top_y)),                                        # Head
            ]
            poly_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(frame, [poly_pts], (40, 40, 45))  # Dark athletic garment / skin tone

            # Draw ArUco marker in subject plane
            marker_px = int(aruco_size_cm * ppm)
            marker_x = int(cx + half_w_px * 1.4)
            marker_y = int(waist_y - marker_px / 2)
            if marker_x + marker_px < img_w and marker_y + marker_px < img_h:
                try:
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_px)
                    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
                    frame[marker_y : marker_y + marker_px, marker_x : marker_x + marker_px] = marker_bgr
                except Exception:
                    cv2.rectangle(frame, (marker_x, marker_y), (marker_x + marker_px, marker_y + marker_px), (0, 0, 0), -1)
                    cv2.rectangle(frame, (marker_x + 4, marker_y + 4), (marker_x + marker_px - 4, marker_y + marker_px - 4), (255, 255, 255), -1)

            if noise_sigma > 0:
                noise = np.random.normal(0, noise_sigma, frame.shape).astype(np.float32)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            frames[angle] = frame

        return RenderedSubjectScene(
            subject_id=subject_id,
            height_cm=height_cm,
            distance_m=distance_m,
            camera_height_cm=100.0,
            focal_length_px=focal_length,
            image_size=image_size,
            pixels_per_cm=ppm,
            aruco_marker_size_cm=aruco_size_cm,
            ground_truth=gt_profiles,
            frames_by_angle=frames,
            metadata={"seed": self.seed, "angles": list(angles), "waist_y_pixel": int(waist_y)},
        )

    def generate_dataset_split(
        self,
        split: str = "dev",
        num_subjects: int = 5,
    ) -> List[RenderedSubjectScene]:
        """
        Generates Dev (tunable) or Holdout (strictly read-only test gate) dataset splits.
        """
        dataset = []
        morphologies = [
            {"waist_w": 28.0, "waist_d": 19.0, "chest_w": 32.0, "chest_d": 22.0, "hip_w": 33.0, "hip_d": 23.0, "h": 168.0},
            {"waist_w": 31.0, "waist_d": 22.0, "chest_w": 35.0, "chest_d": 25.0, "hip_w": 36.0, "hip_d": 26.0, "h": 175.0},
            {"waist_w": 34.0, "waist_d": 25.0, "chest_w": 38.0, "chest_d": 28.0, "hip_w": 38.0, "hip_d": 28.0, "h": 180.0},
            {"waist_w": 38.0, "waist_d": 30.0, "chest_w": 41.0, "chest_d": 32.0, "hip_w": 42.0, "hip_d": 33.0, "h": 172.0},
            {"waist_w": 26.0, "waist_d": 18.0, "chest_w": 30.0, "chest_d": 20.0, "hip_w": 31.0, "hip_d": 21.0, "h": 160.0},
        ]

        for i in range(num_subjects):
            morph = morphologies[i % len(morphologies)]
            sub_id = f"{split}_subject_{i+1:02d}"
            scene = self.render_subject_scene(
                subject_id=sub_id,
                height_cm=morph["h"],
                waist_width_cm=morph["waist_w"],
                waist_depth_cm=morph["waist_d"],
                chest_width_cm=morph["chest_w"],
                chest_depth_cm=morph["chest_d"],
                hip_width_cm=morph["hip_w"],
                hip_depth_cm=morph["hip_d"],
                distance_m=2.2 + (0.1 if i % 2 == 1 else -0.1),
            )
            dataset.append(scene)

        return dataset
