"""
ANTIGRAVITY Evaluation & Test Harness Package.
Provides multi-tier benchmark suites, procedural digital twin generation,
and machine-readable artifact generation.
"""

from eval.synthetic_generator import DigitalTwinGenerator, GroundTruthProfile
from eval.tiers import EvaluationSuite, TierResult

__all__ = [
    "DigitalTwinGenerator",
    "GroundTruthProfile",
    "EvaluationSuite",
    "TierResult",
]
