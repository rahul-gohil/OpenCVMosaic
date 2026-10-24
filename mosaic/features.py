"""Local feature detection with a bounded keypoint budget.

The paper bounds the number of retained keypoints per frame (``N_max``) by
keeping only the strongest detector responses, which is what makes the
matching and the robust estimation cost predictable on a CPU.  OpenCV's SIFT
implementation exposes that budget directly through the ``nfeatures``
constructor argument, so the pruning happens inside the detector instead of
being emulated afterwards.

SIFT is the detector of the paper (Table 1); its scale space makes it robust
to the scale variation between two frames of a flight, and its 128 dimensional
descriptor is what the ratio test of Lowe is defined for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import DetectorConfig


#: The paper reports its keypoint budget ``N_max`` for frames of this size
#: (1920 x 1080).  The budget is scaled with the area of the frame, so that
#: the *density* of the features - and with it the reliability of the
#: registration - stays the same whatever the resolution of the imagery is.
REFERENCE_RESOLUTION = (1920, 1080)

#: Upper bound of the scaled budget, so that the cost of the matching and of
#: the robust estimation stays bounded on very large frames.
MAX_FEATURES = 3000


@dataclass
class FeatureSet:
    """Keypoints and descriptors of a single frame.

    Attributes
    ----------
    points:
        ``(N, 2)`` array of keypoint positions, in source image pixels.
    descriptors:
        ``(N, D)`` descriptor matrix, or ``None`` when no feature was found.
    frame_ids:
        For the features of the mosaic store: the index of the frame each
        feature was observed in.
    """

    points: np.ndarray
    descriptors: Optional[np.ndarray]
    detection_time: float = 0.0
    frame_ids: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def subset(self, mask: np.ndarray) -> "FeatureSet":
        """Return the features selected by a boolean mask."""
        mask = np.asarray(mask, bool)
        return FeatureSet(
            points=self.points[mask],
            descriptors=(
                None if self.descriptors is None else self.descriptors[mask]
            ),
            frame_ids=(
                None if self.frame_ids is None else self.frame_ids[mask]
            ),
        )


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single channel grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class FeatureExtractor:
    """Builds the SIFT detector/descriptor used by the registration stage."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._detector = cv2.SIFT_create(
            nfeatures=config.n_features,
            nOctaveLayers=config.n_octave_layers,
            contrastThreshold=config.contrast_threshold,
            edgeThreshold=config.edge_threshold,
            sigma=config.sigma,
        )

    # ------------------------------------------------------------------- API
    def extract(self, image: np.ndarray, budget: Optional[int] = None) -> FeatureSet:
        """Detect and describe the features of ``image``.

        ``budget`` overrides the keypoint budget that is otherwise derived
        from the size of the image.
        """
        gray = to_gray(image)
        if budget is None:
            budget = self._budget(gray.shape[1], gray.shape[0])
        if budget != self._detector.getNFeatures():
            self._detector.setNFeatures(budget)

        keypoints, descriptors = self._detector.detectAndCompute(gray, None)

        if keypoints is None or len(keypoints) == 0:
            return FeatureSet(
                points=np.empty((0, 2), np.float32),
                descriptors=None,
            )

        return FeatureSet(
            points=np.float32([kp.pt for kp in keypoints]),
            descriptors=descriptors,
        )

    def _budget(self, width: int, height: int) -> int:
        """Return the keypoint budget for a frame of the given size."""
        reference = REFERENCE_RESOLUTION[0] * REFERENCE_RESOLUTION[1]
        scaled = self.config.n_features * (width * height) / float(reference)
        return int(min(max(round(scaled), self.config.n_features), MAX_FEATURES))
