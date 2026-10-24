"""Configuration objects for the incremental orthomosaic pipeline.

Every tunable of the pipeline lives here so that a run is fully described by a
single, serialisable object.  The defaults follow the configuration reported in
the paper "Real-Time 2D Orthomosaic Mapping from UAV Video via Feature-Based
Image Registration" (Appl. Sci. 2026, 16, 2133):

* SIFT with ``nOctaveLayers = 3``, ``contrastThreshold = 0.04``,
  ``edgeThreshold = 10`` and ``sigma = 1.6`` (Table 1 of the paper).
* A keypoint budget ``N_max`` realised by keeping only the strongest detector
  responses (Table 2 reports 500 retained keypoints per frame).
* FLANN KNN matching combined with a ratio test (Table 3).
* RANSAC based geometric verification, followed by a re-estimation of the
  homography on the inlier set only (Sections 3.4 and 3.5).
* Sequential homography accumulation ``G_{k+1} = G_k @ H_{k->k+1}`` (Eq. 6).
* A mosaic canvas that grows as the warped frames extend beyond the current
  extent (Section 3.6).

The values are deliberately expressed in *image pixels* so that they stay
meaningful independently of the resolution the sequence is processed at.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class DetectorConfig:
    """SIFT detector/descriptor configuration.

    ``n_features`` is the keypoint budget ``N_max`` of the paper: only the
    strongest ``n_features`` detector responses are retained, which bounds the
    dominant cost of matching and of the robust estimation.
    """

    n_features: int = 500

    # --- SIFT (paper Table 1) -------------------------------------------------
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: int = 10
    sigma: float = 1.6


@dataclass
class MatcherConfig:
    """Descriptor matching configuration."""

    #: Lowe's ratio test threshold applied to the two nearest neighbours.
    ratio: float = 0.7
    #: Thresholds tried in order; the strictest one that yields enough
    #: correspondences is used.  The ratio test is only a pre-filter, the
    #: geometric verification decides, so a scene full of similar structures
    #: must not be rejected outright.
    ratio_ladder: Tuple[float, ...] = (0.7, 0.8, 0.9, 0.95)
    flann_trees: int = 5
    flann_checks: int = 50


@dataclass
class MosaicStoreConfig:
    """Configuration of the incremental feature store of the mosaic.

    The store holds the SIFT features of the mosaic, expressed in mosaic
    coordinates, so that an incoming frame can be registered against the
    *whole* mosaic instead of against the previous frame only.
    """

    #: Minimum distance, in source pixels, between two stored features.  A new
    #: observation of a ground location replaces the previous one, which keeps
    #: one descriptor per location (duplicated descriptors would break Lowe's
    #: ratio test) and keeps the store small.
    min_distance: float = 40.0
    #: Number of recent frames whose features are kept as a fallback
    #: reference when no prediction is available.
    window: int = 4
    #: Margin, in source pixels, around the predicted position of a frame
    #: that is used to select the reference features of the mosaic.  The
    #: margin has to absorb the residual misalignment of the prediction.
    query_margin: float = 1500.0
    #: Upper bound on the number of stored features.  When it is reached, the
    #: features of the oldest frame are dropped.
    max_features: int = 50000


@dataclass
class RegistrationConfig:
    """Robust geometric verification configuration."""

    #: ``mosaic`` registers the incoming frame against the accumulated mosaic,
    #: which keeps the transform free of accumulated drift; ``frame`` registers
    #: it against the previous frame and accumulates the transforms as in
    #: Eq. (6) of the paper.
    mode: str = "mosaic"
    #: Number of recent frames whose correspondences describe the *relative*
    #: motion of the incoming frame.  The correspondences with the frames
    #: further back describe where the frame sits over the whole mosaic, and
    #: they are what grounds it.
    grounding_window: int = 6
    #: Minimum number of inliers required from the grounding correspondences.
    #: There are only a handful of them, so the bar is much lower than for the
    #: relative motion.
    grounding_min_inliers: int = 8
    #: Minimum inlier ratio of the grounding correspondences.
    grounding_min_inlier_ratio: float = 0.5
    #: Maximum displacement the grounding model is allowed to correct, as a
    #: fraction of the diagonal of the frame.  It guards against a grounding
    #: model that is geometrically consistent but describes a different part
    #: of the mosaic.
    grounding_max_shift: float = 0.15
    #: Maximum reprojection error of the grounding model, in source pixels.
    grounding_max_residual: float = 2.0
    #: RANSAC reprojection error threshold, in source pixels.
    ransac_threshold: float = 5.0
    max_iters: int = 5000
    confidence: float = 0.995
    #: Minimum number of RANSAC inliers required to accept a frame.
    min_inliers: int = 20
    #: Minimum inlier ratio (inliers / tentative matches) required to accept.
    #: It has to be low: on a repetitive scene the ratio test has to be
    #: relaxed, and the true correspondences are then a small minority of the
    #: candidates.  The number of inliers is what carries the decision.
    min_inlier_ratio: float = 0.10
    #: Re-estimate the homography on the inlier set (paper, Section 3.5).
    refine_with_inliers: bool = True


@dataclass
class BlendingConfig:
    """Mosaic blending configuration."""

    #: ``overwrite`` is the blending rule the paper implements (Section 3.6):
    #: the pixels of the latest warped frame replace the existing ones in the
    #: overlap regions.  ``feather`` mixes the two images with a ramp over the
    #: border of the warped frame.
    method: str = "overwrite"
    #: Width, in pixels, of the transition band of the feather blend.
    ramp: int = 32


@dataclass
class CanvasConfig:
    """Mosaic canvas configuration."""

    #: Resolution of the mosaic relative to the source images.
    scale: float = 1.0
    #: Extra pixels added around the canvas whenever it has to grow, to avoid
    #: reallocating on every single step.
    grow_margin: int = 256
    #: Interpolation used when warping a frame onto the canvas.
    interpolation: int = 1  # cv2.INTER_LINEAR
    #: Upper bound on the number of pixels of the canvas.  The canvas needs
    #: about four bytes per pixel (three for the colour, one for the mask), so
    #: this bound directly translates into a memory budget.  ``0`` disables
    #: the check.
    max_pixels: int = 0


@dataclass
class OutputConfig:
    """Artefact writing configuration."""

    root: str = "outputs"
    run_name: str | None = None
    #: Write a downscaled preview of the mosaic after every update.
    save_preview: bool = True
    preview_scale: float = 0.1
    #: Write the final mosaic when the run terminates.
    save_final: bool = True
    #: Dump the per step timings as a CSV file.
    save_timings: bool = True
    #: Write the match / inlier visualisations for every step.
    save_debug: bool = False
    jpeg_quality: int = 90


@dataclass
class MosaicConfig:
    """Top level configuration of a mosaic run."""

    dataset_dir: str
    image_pattern: str = "*.JPG"
    #: Index of the first / last image of the sequence (inclusive, 0 based).
    start: int | None = None
    end: int | None = None
    #: Try to re-link a frame to the last accepted frame when the sequential
    #: registration fails, instead of dropping it.
    relink_on_failure: bool = True

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    store: MosaicStoreConfig = field(default_factory=MosaicStoreConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary description of the configuration."""
        return asdict(self)

    def summary(self) -> str:
        """Return a short, human readable one-liner describing the run."""
        return (
            f"detector=sift(N_max={self.detector.n_features}) "
            f"matcher=flann(ratio={self.matcher.ratio}) "
            f"registration={self.registration.mode} "
            f"ransac={self.registration.ransac_threshold}px "
            f"blend={self.blending.method} canvas_scale={self.canvas.scale}"
        )
