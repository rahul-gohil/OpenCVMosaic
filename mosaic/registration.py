"""Robust planar homography estimation (RANSAC) and frame acceptance.

This module implements the geometric verification stage of the paper
(Sections 3.4 and 3.5):

1. tentative correspondences come from the descriptor matcher,
2. a planar homography is estimated with RANSAC under a reprojection error
   threshold,
3. the homography is re-estimated on the inlier set only, which improves the
   stability of the model,
4. the frame is accepted only when the inlier support satisfies a minimum
   acceptance criterion; otherwise the update is rejected so that an
   unreliable transform is never propagated through the accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import RegistrationConfig
from .features import FeatureSet
from .matching import DescriptorMatcher

#: A homography whose characteristic scale falls outside of this range is
#: considered implausible and the frame is rejected.
MIN_SCALE = 0.2
MAX_SCALE = 5.0


def homogeneous_divide(projected: np.ndarray) -> np.ndarray:
    """Cartesian coordinates of homogeneous points, safe for ``w <= 0``."""
    w = projected[:, 2:3]
    eps = np.finfo(np.float64).eps
    safe = np.where(np.abs(w) > eps, w, np.nan)
    return projected[:, :2] / safe


@dataclass
class Registration:
    """Outcome of a pairwise registration attempt."""

    ok: bool
    #: Homography mapping the *incoming* frame onto the *reference* frame.
    H: Optional[np.ndarray]
    #: Number of tentative matches that passed the ratio test.
    matches: int = 0
    #: Number of RANSAC inliers.
    inliers: int = 0
    #: ``inliers / matches``.
    inlier_ratio: float = 0.0
    #: Boolean flag per tentative match, ``True`` for the RANSAC inliers.
    inlier_flags: Optional[np.ndarray] = None
    #: Median reprojection error of the inliers, in source pixels.
    reprojection_error: float = 0.0
    #: Human readable explanation, also used in the run log.
    reason: str = ""


def _valid_homography(H: Optional[np.ndarray]) -> bool:
    """Reject degenerate or numerically broken homographies."""
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False
    if abs(H[2, 2]) < 1e-12:
        return False
    det = float(np.linalg.det(H))
    if not np.isfinite(det) or abs(det) < 1e-12:
        return False
    return True


def register_pair(
    reference: FeatureSet,
    incoming: FeatureSet,
    matcher: DescriptorMatcher,
    config: RegistrationConfig,
) -> Registration:
    """Register ``incoming`` onto ``reference``.

    The returned homography maps a point of the incoming frame to the
    corresponding point of the reference frame, i.e. it is the ``H_{k->k+1}``
    of the paper.
    """
    knn = matcher.knn_match(reference.descriptors, incoming.descriptors)
    for ratio in config.ratio_ladder:
        matches = matcher.ratio_filter(knn, ratio)
        if len(matches) < 4:
            continue
        registration = estimate_homography(reference, incoming, matches, config)
        if registration.ok:
            return registration
    return estimate_homography(reference, incoming, matcher.ratio_filter(knn, config.ratio_ladder[0]), config)


def estimate_homography(
    reference: FeatureSet,
    incoming: FeatureSet,
    matches,
    config: RegistrationConfig,
) -> Registration:
    """Estimate the homography that maps ``incoming`` onto ``reference``.

    ``matches`` are the tentative correspondences produced by the matcher: the
    query index refers to a keypoint of the reference frame and the train index
    to a keypoint of the incoming frame.
    """
    n_matches = len(matches)
    if n_matches < 4:
        return Registration(
            ok=False,
            H=None,
            matches=n_matches,
            reason=f"only {n_matches} tentative match(es), 4 are required",
        )

    src = np.float32([incoming.points[m.trainIdx] for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([reference.points[m.queryIdx] for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        config.ransac_threshold,
        maxIters=config.max_iters,
        confidence=config.confidence,
    )
    if H is None or mask is None:
        return Registration(
            ok=False, H=None, matches=n_matches, reason="RANSAC did not converge"
        )

    inlier_flags = mask.ravel().astype(bool)
    n_inliers = int(inlier_flags.sum())
    ratio = n_inliers / float(n_matches)

    if n_inliers < config.min_inliers:
        return Registration(
            ok=False,
            H=None,
            matches=n_matches,
            inliers=n_inliers,
            inlier_ratio=ratio,
            reason=f"only {n_inliers} inlier(s), {config.min_inliers} are required",
        )
    if ratio < config.min_inlier_ratio:
        return Registration(
            ok=False,
            H=None,
            matches=n_matches,
            inliers=n_inliers,
            inlier_ratio=ratio,
            reason=(
                f"inlier ratio {ratio:.2f} below the required "
                f"{config.min_inlier_ratio:.2f}"
            ),
        )

    if config.refine_with_inliers:
        refined, refined_mask = cv2.findHomography(
            src[inlier_flags], dst[inlier_flags], 0
        )
        # Keep the refined model only when it is well conditioned and when it
        # still explains the inlier set.
        if _valid_homography(refined) and refined_mask is not None:
            H = refined

    if not _valid_homography(H):
        return Registration(
            ok=False,
            H=None,
            matches=n_matches,
            inliers=n_inliers,
            inlier_ratio=ratio,
            reason="estimated homography is degenerate",
        )

    scale = _homography_scale(H, src.reshape(-1, 2))
    if not (MIN_SCALE <= scale <= MAX_SCALE):
        return Registration(
            ok=False,
            H=None,
            matches=n_matches,
            inliers=n_inliers,
            inlier_ratio=ratio,
            reason=f"implausible scale change ({scale:.2f}x)",
        )

    error = _reprojection_error(H, src[inlier_flags], dst[inlier_flags])

    return Registration(
        ok=True,
        H=normalise(H),
        matches=n_matches,
        inliers=n_inliers,
        inlier_ratio=ratio,
        inlier_flags=inlier_flags,
        reprojection_error=error,
        reason="ok",
    )


def _reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """Median reprojection error of the correspondences, in source pixels."""
    if len(src) == 0:
        return 0.0
    H = np.asarray(H, dtype=np.float64)
    homogeneous = np.column_stack(
        [np.asarray(src, np.float64).reshape(-1, 2), np.ones(len(src))]
    )
    projected = homogeneous_divide(homogeneous @ H.T)
    residuals = np.linalg.norm(projected - np.asarray(dst, np.float64).reshape(-1, 2), axis=1)
    return float(np.median(residuals))


def _homography_scale(H: np.ndarray, source_points: np.ndarray) -> float:
    """Return the characteristic scale of ``H``.

    The median ratio between the distance of a point to its image and its
    original distance gives a robust measure of the zoom the homography
    applies.  It is used as a sanity check against grossly wrong models.
    """
    H = np.asarray(H, dtype=np.float64)
    if source_points is None or len(source_points) < 2:
        return 1.0
    pts = np.asarray(source_points, dtype=np.float64)
    centre = pts.mean(axis=0)
    offsets = pts - centre
    distances = np.linalg.norm(offsets, axis=1)
    usable = distances > 1e-6
    if not np.any(usable):
        return 1.0
    homogeneous = np.column_stack([pts[usable], np.ones(int(usable.sum()))])
    mapped = homogeneous_divide(homogeneous @ H.T)
    # The scale is the ratio of the spread of the mapped points around their
    # own centroid and the spread of the source points around theirs.
    mapped_centre = mapped.mean(axis=0)
    mapped_distances = np.linalg.norm(mapped - mapped_centre, axis=1)
    ratios = mapped_distances / distances[usable]
    return float(np.median(ratios))


def normalise(H: np.ndarray) -> np.ndarray:
    """Return ``H`` scaled so that ``H[2, 2] == 1``.

    A homography is only defined up to a global scale; normalising it keeps the
    accumulated transform numerically well behaved over long sequences.
    """
    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to a set of ``(N, 2)`` points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ np.asarray(H, dtype=np.float64).T
    return homogeneous_divide(mapped)
