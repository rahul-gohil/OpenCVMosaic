"""Incremental 2D orthomosaic construction from an ordered image sequence.

The package implements a CPU-friendly, feature based incremental mosaicking
pipeline that follows the processing procedure described in the paper
"Real-Time 2D Orthomosaic Mapping from UAV Video via Feature-Based Image
Registration" (Appl. Sci. 2026, 16, 2133), restricted to still image
sequences.

The pipeline is organised as a set of small, single purpose modules:

``config``       every tunable of the pipeline, in one serialisable object.
``dataset``      ordered discovery / loading of the source images.
``features``     local feature detection with a bounded keypoint budget.
``matching``     descriptor matching with a ratio test.
``registration`` robust planar homography estimation (RANSAC) + acceptance.
``transforms``   accumulation of inter-frame homographies into mosaic frame.
``blending``     feather or overwrite blending (no multi-band).
``canvas``       growable mosaic canvas and warped image placement.
``pipeline``     the incremental stitching loop and its runtime accounting.
``output``       artefact writing (canvas snapshots, timings, final mosaic).
"""

from .config import BlendingConfig, CanvasConfig, DetectorConfig, MatcherConfig, MosaicConfig
from .config import MosaicStoreConfig, OutputConfig, RegistrationConfig
from .pipeline import RunReport, StitchingPipeline
from .merge import MosaicMerger
from .chunker import ChunkRunner
from .store import MosaicFeatureStore

__all__ = [
    "BlendingConfig",
    "CanvasConfig",
    "ChunkRunner",
    "DetectorConfig",
    "MatcherConfig",
    "MosaicConfig",
    "MosaicFeatureStore",
    "MosaicMerger",
    "MosaicStoreConfig",
    "OutputConfig",
    "RegistrationConfig",
    "RunReport",
    "StitchingPipeline",
]
