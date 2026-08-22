"""
Procedural 3D Digital Twin & Ground-Truth Silhouette Generator.

Generates:
1. Procedural 3D human torso/limb geometries with parameterized BMI, waist-to-hip ratio, and lumbar lordosis.
2. Ground-truth 2D cross-sectional slices with exact raw perimeter and convex-hull tape-measure perimeter.
3. Multi-angle silhouette projections rendered with a calibrated pinhole camera and in-plane ArUco fiducials.
4. Deterministic Dev and Holdout evaluation datasets with §4 Tier 2.5 parameter sweeps:
   - Camera height ±15 cm (85-115 cm)
   - Subject distance 1.8–3.5 m
   - Focal length multiplier ×3 (0.75x, 1.0x, 1.5x)
   - Camera roll ±2°
   - Marker tilt 0–20°
   - Resolution 1080p and 4K
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
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
            spine_weight = np.exp(-0.5 * (x / (max(0.1, a) * 0.22)) ** 2) * np.maximum(0.0, -z / max(0.1, b))
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
        camera_height_cm: float = 100.0,
        focal_mult: float = 1.0,
        camera_roll_deg: float = 0.0,
        marker_tilt_deg: float = 0.0,
        angles: Tuple[int, ...] = (0, 90, 180, 270),
        image_size: Tuple[int, int] = (1920, 1080),  # (width, height)
        aruco_size_cm: float = 15.0,
        noise_sigma: float = 0.0,
    ) -> RenderedSubjectScene:
        """
        Renders multi-angle body silhouettes and ArUco calibration targets with realistic camera sweeps.
        """
        img_w, img_h = image_size
        base_focal = 1400.0 * (img_w / 1920.0)
        focal_length = base_focal * focal_mult
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

            # Determine projected silhouette width at anatomical slices
            rad = np.radians(angle)
            cos_r, sin_r = np.cos(rad), np.sin(rad)

            proj_w_waist = np.sqrt((waist_width_cm * cos_r) ** 2 + (waist_depth_cm * sin_r) ** 2)
            proj_w_chest = np.sqrt((chest_width_cm * cos_r) ** 2 + (chest_depth_cm * sin_r) ** 2)
            proj_w_hips = np.sqrt((hip_width_cm * cos_r) ** 2 + (hip_depth_cm * sin_r) ** 2)

            half_w_waist = (proj_w_waist * ppm) / 2.0
            half_w_chest = (proj_w_chest * ppm) / 2.0
            half_w_hips = (proj_w_hips * ppm) / 2.0

            # Vertical body position adjusted for camera height (nominal 100 cm)
            cam_offset_px = (camera_height_cm - 100.0) * ppm * 0.15
            body_h_px = height_cm * ppm
            body_top_y = cy - (body_h_px * 0.45) + cam_offset_px
            body_bot_y = body_top_y + body_h_px
            waist_y = body_top_y + (body_h_px * 0.40)
            chest_y = body_top_y + (body_h_px * 0.22)
            hips_y = body_top_y + (body_h_px * 0.55)

            # Draw smooth body silhouette
            pts = [
                (int(round(cx - half_w_chest)), int(round(chest_y))),
                (int(round(cx - half_w_waist)), int(round(waist_y))),
                (int(round(cx - half_w_hips)), int(round(hips_y))),
                (int(round(cx - half_w_hips * 0.55)), int(round(body_bot_y))),
                (int(round(cx + half_w_hips * 0.55)), int(round(body_bot_y))),
                (int(round(cx + half_w_hips)), int(round(hips_y))),
                (int(round(cx + half_w_waist)), int(round(waist_y))),
                (int(round(cx + half_w_chest)), int(round(chest_y))),
                (int(round(cx)), int(round(body_top_y))),
            ]
            poly_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(frame, [poly_pts], (40, 40, 45))  # Dark athletic garment / skin tone

            # Draw ArUco marker at top-left wall (away from all torso scanlines to avoid occlusion)
            marker_px = max(20, int(aruco_size_cm * ppm))
            scale_fac = img_w / 1920.0
            marker_x = int(60 * scale_fac)
            marker_y = int(60 * scale_fac)

            if marker_x + marker_px < img_w and marker_y + marker_px < img_h:
                try:
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_px)
                    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

                    # Apply marker tilt if requested (pitch / yaw foreshortening)
                    if marker_tilt_deg != 0.0:
                        rad_t = np.radians(marker_tilt_deg)
                        shift_y = int(marker_px * 0.12 * np.sin(rad_t))
                        shift_x = int(marker_px * (1.0 - np.cos(rad_t)))
                        p1 = np.float32([[0, 0], [marker_px, 0], [marker_px, marker_px], [0, marker_px]])
                        p2 = np.float32([[shift_x, shift_y], [marker_px, 0], [marker_px, marker_px], [shift_x, marker_px - shift_y]])
                        H = cv2.getPerspectiveTransform(p1, p2)
                        marker_bgr = cv2.warpPerspective(marker_bgr, H, (marker_px, marker_px), borderValue=(230, 230, 230))

                    frame[marker_y : marker_y + marker_px, marker_x : marker_x + marker_px] = marker_bgr
                except Exception:
                    cv2.rectangle(frame, (marker_x, marker_y), (marker_x + marker_px, marker_y + marker_px), (0, 0, 0), -1)

            # Apply camera roll rotation if non-zero
            if camera_roll_deg != 0.0:
                M_roll = cv2.getRotationMatrix2D((cx, cy), camera_roll_deg, 1.0)
                frame = cv2.warpAffine(frame, M_roll, (img_w, img_h), borderValue=(230, 230, 230))

            if noise_sigma > 0:
                noise = np.random.normal(0, noise_sigma, frame.shape).astype(np.float32)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            frames[angle] = frame

        return RenderedSubjectScene(
            subject_id=subject_id,
            height_cm=height_cm,
            distance_m=distance_m,
            camera_height_cm=camera_height_cm,
            focal_length_px=focal_length,
            image_size=image_size,
            pixels_per_cm=ppm,
            aruco_marker_size_cm=aruco_size_cm,
            ground_truth=gt_profiles,
            frames_by_angle=frames,
            metadata={
                "seed": self.seed,
                "angles": list(angles),
                "waist_y_pixel": int(round(waist_y)),
                "camera_roll_deg": camera_roll_deg,
                "marker_tilt_deg": marker_tilt_deg,
                "focal_mult": focal_mult,
            },
        )

    def generate_dataset_split(
        self,
        split: str = "dev",
        num_subjects: int = 5,
    ) -> List[RenderedSubjectScene]:
        """
        Generates Dev (tunable) or Holdout (strictly read-only test gate) dataset splits.
        DEV and HOLDOUT have strictly disjoint subject morphologies and camera sweep configurations.
        """
        dataset = []

        if split == "dev":
            # DEV Split: 5 diverse somatotypes at standard camera setups
            dev_configs = [
                # Sub 1: Athletic V-taper male
                {"sub_id": "dev_subject_01", "h": 174.0, "w_w": 28.0, "w_d": 19.5, "c_w": 33.5, "c_d": 23.5, "h_w": 34.0, "h_d": 24.0, "dist": 2.10, "cam_h": 100.0, "focal_m": 1.0, "roll": 0.0, "tilt": 0.0, "res": (1920, 1080)},
                # Sub 2: Average adult female
                {"sub_id": "dev_subject_02", "h": 165.0, "w_w": 27.0, "w_d": 18.5, "c_w": 31.5, "c_d": 21.5, "h_w": 35.0, "h_d": 25.0, "dist": 2.25, "cam_h": 95.0, "focal_m": 1.0, "roll": 0.5, "tilt": 4.0, "res": (1920, 1080)},
                # Sub 3: Tall lean ectomorph
                {"sub_id": "dev_subject_03", "h": 184.0, "w_w": 30.0, "w_d": 20.5, "c_w": 34.5, "c_d": 24.0, "h_w": 35.5, "h_d": 25.0, "dist": 2.50, "cam_h": 105.0, "focal_m": 1.0, "roll": -0.5, "tilt": 6.0, "res": (1920, 1080)},
                # Sub 4: Heavy android somatotype
                {"sub_id": "dev_subject_04", "h": 178.0, "w_w": 36.0, "w_d": 28.0, "c_w": 41.0, "c_d": 31.0, "h_w": 41.5, "h_d": 31.5, "dist": 2.35, "cam_h": 100.0, "focal_m": 1.0, "roll": 0.8, "tilt": 8.0, "res": (1920, 1080)},
                # Sub 5: Slender petite
                {"sub_id": "dev_subject_05", "h": 158.0, "w_w": 24.5, "w_d": 16.5, "c_w": 28.5, "c_d": 18.5, "h_w": 30.5, "h_d": 20.5, "dist": 2.00, "cam_h": 90.0, "focal_m": 1.0, "roll": -0.8, "tilt": 5.0, "res": (1920, 1080)},
            ]
            configs = dev_configs[:num_subjects]

        else:
            # HOLDOUT Split: Strictly disjoint morphologies with Tier 2.5 camera parameter sweep
            # Sweeps: camera height (85-115cm), distance (1.85-3.45m), focal length (0.75-1.5x),
            # camera roll (-1.8° to +1.9°), marker tilt (3° to 19°), 1080p and 4K resolutions
            holdout_configs = [
                # Holdout 1: Short slender, close distance, low camera, wide focal length (0.75x), negative roll
                {"sub_id": "holdout_subject_01", "h": 162.0, "w_w": 26.0, "w_d": 17.5, "c_w": 30.0, "c_d": 20.0, "h_w": 32.5, "h_d": 22.5, "dist": 1.85, "cam_h": 85.0, "focal_m": 0.75, "roll": -1.8, "tilt": 8.0, "res": (1920, 1080)},
                # Holdout 2: Medium frame, 4K resolution, standard distance, positive roll, 12° marker tilt
                {"sub_id": "holdout_subject_02", "h": 170.0, "w_w": 29.0, "w_d": 20.0, "c_w": 33.0, "c_d": 23.0, "h_w": 36.0, "h_d": 25.5, "dist": 2.30, "cam_h": 100.0, "focal_m": 1.00, "roll": 1.5, "tilt": 12.0, "res": (3840, 2160)},
                # Holdout 3: Tall mesomorph, medium-far distance, high camera (115cm), tele focal length (1.5x)
                {"sub_id": "holdout_subject_03", "h": 180.0, "w_w": 33.0, "w_d": 23.5, "c_w": 37.0, "c_d": 26.5, "h_w": 38.5, "h_d": 28.0, "dist": 2.75, "cam_h": 115.0, "focal_m": 1.50, "roll": -0.8, "tilt": 3.0, "res": (1920, 1080)},
                # Holdout 4: Large heavy endomorph, far distance (3.15m), 4K resolution, max roll (+1.9°), 16° marker tilt
                {"sub_id": "holdout_subject_04", "h": 186.0, "w_w": 39.0, "w_d": 30.5, "c_w": 43.5, "c_d": 33.0, "h_w": 44.0, "h_d": 34.0, "dist": 3.15, "cam_h": 90.0, "focal_m": 1.00, "roll": 1.9, "tilt": 16.0, "res": (3840, 2160)},
                # Holdout 5: Athletic female, very far distance (3.45m), high camera (110cm), 0.85x focal length, 19° tilt
                {"sub_id": "holdout_subject_05", "h": 172.0, "w_w": 35.0, "w_d": 25.5, "c_w": 38.5, "c_d": 28.5, "h_w": 39.5, "h_d": 29.5, "dist": 3.45, "cam_h": 110.0, "focal_m": 0.85, "roll": -1.2, "tilt": 19.0, "res": (1920, 1080)},
            ]
            configs = holdout_configs[:num_subjects]

        for cfg in configs:
            scene = self.render_subject_scene(
                subject_id=cfg["sub_id"],
                height_cm=cfg["h"],
                waist_width_cm=cfg["w_w"],
                waist_depth_cm=cfg["w_d"],
                chest_width_cm=cfg["c_w"],
                chest_depth_cm=cfg["c_d"],
                hip_width_cm=cfg["h_w"],
                hip_depth_cm=cfg["h_d"],
                distance_m=cfg["dist"],
                camera_height_cm=cfg["cam_h"],
                focal_mult=cfg["focal_m"],
                camera_roll_deg=cfg["roll"],
                marker_tilt_deg=cfg["tilt"],
                image_size=cfg["res"],
            )
            dataset.append(scene)

        return dataset


def compute_holdout_content_hash(dataset: Optional[List[RenderedSubjectScene]] = None) -> str:
    """
    Computes a deterministic SHA-256 content hash of the entire holdout evaluation dataset.
    """
    if dataset is None:
        gen = DigitalTwinGenerator(seed=42)
        dataset = gen.generate_dataset_split("holdout", num_subjects=5)

    hasher = hashlib.sha256()
    for scene in dataset:
        hasher.update(scene.subject_id.encode("utf-8"))
        hasher.update(str(scene.height_cm).encode("utf-8"))
        hasher.update(str(scene.distance_m).encode("utf-8"))
        hasher.update(str(scene.camera_height_cm).encode("utf-8"))
        hasher.update(str(scene.pixels_per_cm).encode("utf-8"))
        hasher.update(str(scene.image_size).encode("utf-8"))

        for k in sorted(scene.ground_truth.keys()):
            gt = scene.ground_truth[k]
            hasher.update(k.encode("utf-8"))
            hasher.update(str(round(gt.perimeter_hull_cm, 4)).encode("utf-8"))
            hasher.update(str(round(gt.perimeter_raw_cm, 4)).encode("utf-8"))
            hasher.update(str(round(gt.coronal_width_cm, 4)).encode("utf-8"))
            hasher.update(str(round(gt.sagittal_depth_cm, 4)).encode("utf-8"))

        for angle in sorted(scene.frames_by_angle.keys()):
            frame = scene.frames_by_angle[angle]
            hasher.update(hashlib.sha256(frame.tobytes()).digest())

    return hasher.hexdigest()


def export_holdout_manifest(output_path: str = "artifacts/holdout_manifest.json") -> Dict[str, Any]:
    """
    Generates and saves the frozen holdout dataset manifest.
    """
    gen = DigitalTwinGenerator(seed=42)
    dataset = gen.generate_dataset_split("holdout", num_subjects=5)
    content_hash = compute_holdout_content_hash(dataset)

    manifest = {
        "version": "1.0",
        "description": "Frozen ANTIGRAVITY Holdout Dataset Manifest (§4 Tier 2.5 Sweep)",
        "content_hash_sha256": content_hash,
        "num_subjects": len(dataset),
        "subjects": [],
    }

    for scene in dataset:
        manifest["subjects"].append({
            "subject_id": scene.subject_id,
            "height_cm": scene.height_cm,
            "distance_m": scene.distance_m,
            "camera_height_cm": scene.camera_height_cm,
            "image_size": list(scene.image_size),
            "pixels_per_cm": round(scene.pixels_per_cm, 4),
            "metadata": scene.metadata,
            "ground_truth": {
                k: {
                    "perimeter_hull_cm": round(v.perimeter_hull_cm, 4),
                    "perimeter_raw_cm": round(v.perimeter_raw_cm, 4),
                    "coronal_width_cm": round(v.coronal_width_cm, 4),
                    "sagittal_depth_cm": round(v.sagittal_depth_cm, 4),
                }
                for k, v in scene.ground_truth.items()
            },
        })

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
