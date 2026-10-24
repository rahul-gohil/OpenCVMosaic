"""Seamless mosaic blending.

The mosaic is built by warping every incoming frame onto a common canvas and
merging it with the content that is already there.  Two strategies are
available:

``overwrite``
    The pixels of the latest warped frame replace the existing ones in the
    overlap regions.  It is the blending rule of the paper (Section 3.6) and
    the cheapest one.  A misregistration stays visible as a seam, but nothing
    of the previous frame survives, so nothing can ghost.

``feather``
    The two images are mixed with a ramp that grows over the last
    ``ramp`` pixels of the warped frame.  It hides the seam of the overwrite
    at the cost of mixing the two images where they overlap, which is only
    visible when they are misregistered.

A multi-band (Laplacian pyramid) blend is deliberately *not* offered: it
mixes the high frequencies of both images locally, so on a scene with
parallax - trees, buildings, anything that is not a plane - the two views of
the same object survive side by side and the mosaic shows ghosting.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import BlendingConfig


def feather_blend(
    base_image: np.ndarray,
    base_mask: np.ndarray,
    new_image: np.ndarray,
    new_mask: np.ndarray,
    config: BlendingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend with a ramp that grows over the border of the warped frame.

    All four arrays describe the same region of interest.  The function
    returns the blended region and the mask of the pixels that changed, so
    that the caller only writes back what is needed.
    """
    valid = new_mask > 0
    if not valid.any():
        return base_image, np.zeros(base_mask.shape, np.uint8)

    ramp = max(1, int(config.ramp))
    inside = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(inside / float(ramp), 0.0, 1.0)
    weight[~valid] = 0.0

    blended = base_image.astype(np.float32) * (1.0 - weight[..., None]) + (
        new_image.astype(np.float32) * weight[..., None]
    )
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended, (weight > 0).astype(np.uint8) * 255


def overwrite_blend(
    base_image: np.ndarray,
    base_mask: np.ndarray,
    new_image: np.ndarray,
    new_mask: np.ndarray,
    config: BlendingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """The light-weight baseline of the paper: the newest frame wins."""
    valid = (new_mask > 0).astype(np.uint8)
    if not valid.any():
        return base_image, np.zeros(base_mask.shape, np.uint8)
    blended = base_image.copy()
    cv2.copyTo(new_image, valid, blended)
    return blended, valid * 255


BLENDERS = {
    "feather": feather_blend,
    "overwrite": overwrite_blend,
}


def blend(
    base_image: np.ndarray,
    base_mask: np.ndarray,
    new_image: np.ndarray,
    new_mask: np.ndarray,
    config: BlendingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend two images according to ``config.method``."""
    method = config.method.lower()
    if method not in BLENDERS:
        raise ValueError(f"unknown blending method: {config.method!r}")
    return BLENDERS[method](base_image, base_mask, new_image, new_mask, config)
