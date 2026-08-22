"""
Exercise App - 4-Angle Guided Capture Body Measurement System.
High-precision computer vision and biomechanical perimeter estimation package.
"""

from body_measurement.adversarial_simulator import (
    AdversarialSimulationConfig,
    AdversarialSimulator,
    GroundTruthCrossSection,
    SimulationEvaluationResult,
)
from body_measurement.burst_processor import BurstAngleResult, BurstFrameProcessor
from body_measurement.edge_detection import EdgeSliceResult, SubPixelEdgeDetector
from body_measurement.landmarks import (
    AnatomicalAnchorEngine,
    AnatomicalAnchorResult,
    BodySite,
)
from body_measurement.reconstruction import (
    CrossSectionReconstructor,
    CrossSectionResult,
    ReconstructionMethod,
)
from body_measurement.scaling import ArucoMetricScaler, CalibrationResult
from body_measurement.system import (
    BodyMeasurementSummary,
    BodyMeasurementSystem,
    CaptureAngle,
)

__version__ = "1.0.0"

__all__ = [
    "ArucoMetricScaler",
    "CalibrationResult",
    "AnatomicalAnchorEngine",
    "AnatomicalAnchorResult",
    "BodySite",
    "SubPixelEdgeDetector",
    "EdgeSliceResult",
    "BurstFrameProcessor",
    "BurstAngleResult",
    "CrossSectionReconstructor",
    "CrossSectionResult",
    "ReconstructionMethod",
    "AdversarialSimulator",
    "AdversarialSimulationConfig",
    "GroundTruthCrossSection",
    "SimulationEvaluationResult",
    "BodyMeasurementSystem",
    "BodyMeasurementSummary",
    "CaptureAngle",
]
