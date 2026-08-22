"""
Architectural Import Firewall Test.

Strictly enforces Rule 3:
body_measurement/** must NEVER import anny, clad_body, clad-body, or trimesh.
Those evaluation and ground-truth libraries belong exclusively to eval/ (the ruler)
and must never leak into the production body_measurement library (the measured system).
"""

import ast
import os
from pathlib import Path
import pytest

FORBIDDEN_MODULES = {"anny", "clad_body", "clad-body", "trimesh", "albumentations", "imagecorruptions"}
BODY_MEASUREMENT_DIR = Path("body_measurement")


def get_imported_modules(py_file_path: Path) -> set:
    """Extracts all imported module names from a Python source file using AST."""
    with open(py_file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(py_file_path))

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def test_body_measurement_import_firewall():
    """
    Scans every Python file in body_measurement/ and asserts that NO forbidden
    evaluation/ground-truth libraries are imported.
    """
    assert BODY_MEASUREMENT_DIR.exists(), "body_measurement directory must exist"

    violations = []
    py_files = list(BODY_MEASUREMENT_DIR.rglob("*.py"))
    assert len(py_files) > 0, "body_measurement must contain python files"

    for py_file in py_files:
        imported = get_imported_modules(py_file)
        forbidden_found = imported.intersection(FORBIDDEN_MODULES)
        if forbidden_found:
            violations.append(f"{py_file}: imports forbidden modules {forbidden_found}")

    assert not violations, f"FIREWALL VIOLATION DETECTED:\n" + "\n".join(violations)
