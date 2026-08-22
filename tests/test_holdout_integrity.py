"""
Holdout Evaluation Set Integrity & Anti-Tampering Test.

Enforces that the holdout test set is 100% frozen, read-only, and deterministically matches
the registered SHA-256 content hash in artifacts/holdout_manifest.json.
"""

import json
from pathlib import Path
import pytest
from eval.synthetic_generator import DigitalTwinGenerator, compute_holdout_content_hash

MANIFEST_PATH = Path("artifacts/holdout_manifest.json")


def test_holdout_dataset_integrity():
    """
    Validates that the holdout dataset content hash has not been modified,
    re-seeded, or reshuffled.
    """
    assert MANIFEST_PATH.exists(), f"Holdout manifest missing at {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_hash = manifest["content_hash_sha256"]

    # Re-generate holdout from generator with canonical seed
    gen = DigitalTwinGenerator(seed=42)
    holdout_dataset = gen.generate_dataset_split("holdout", num_subjects=5)

    computed_hash = compute_holdout_content_hash(holdout_dataset)

    assert computed_hash == expected_hash, (
        f"HOLDOUT INTEGRITY VIOLATION: Computed hash {computed_hash} does not match "
        f"frozen manifest hash {expected_hash}."
    )


def test_dev_and_holdout_disjointness():
    """
    Validates that dev and holdout splits have strictly disjoint subject IDs and morphologies.
    """
    gen = DigitalTwinGenerator(seed=42)
    dev_split = gen.generate_dataset_split("dev", num_subjects=5)
    holdout_split = gen.generate_dataset_split("holdout", num_subjects=5)

    dev_ids = {s.subject_id for s in dev_split}
    holdout_ids = {s.subject_id for s in holdout_split}

    assert dev_ids.isdisjoint(holdout_ids), "Dev and Holdout splits must have disjoint subject IDs."

    dev_morphs = {
        (s.height_cm, round(s.ground_truth["waist"].coronal_width_cm, 2), round(s.ground_truth["waist"].sagittal_depth_cm, 2))
        for s in dev_split
    }
    holdout_morphs = {
        (s.height_cm, round(s.ground_truth["waist"].coronal_width_cm, 2), round(s.ground_truth["waist"].sagittal_depth_cm, 2))
        for s in holdout_split
    }

    assert dev_morphs.isdisjoint(holdout_morphs), "Dev and Holdout splits must have disjoint morphologies."
