"""Accumulation of inter-frame homographies into the mosaic frame.

The paper composes the estimated inter-frame transforms sequentially
(Eq. 6)::

    G_0 = I ,   G_{k+1} = G_k @ H_{k->k+1}

``G_k`` maps the pixels of frame ``k`` into the coordinate system of the
mosaic canvas.  Because the first frame defines the canvas frame, ``G_1``
is exactly the ``H_{2->1}`` of the first pair: the second frame is warped onto
the first one and dropped on the canvas.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from .registration import normalise


class TransformChain:
    """Keeps the mosaic-frame transform of every processed frame."""

    def __init__(self) -> None:
        self._transforms: Dict[int, np.ndarray] = {}
        self._order: List[int] = []

    def seed(self, index: int) -> np.ndarray:
        """Register the first frame with the identity transform."""
        identity = np.eye(3, dtype=np.float64)
        self._transforms[index] = identity
        self._order.append(index)
        return identity

    def accumulate(self, index: int, parent_index: int, H: np.ndarray) -> np.ndarray:
        """Return ``G_index = G_parent @ H`` and record it.

        ``H`` maps the frame ``index`` onto its parent frame ``parent_index``.
        """
        parent = self._transforms.get(parent_index)
        if parent is None:
            raise KeyError(f"no accumulated transform for frame {parent_index}")
        G = normalise(parent @ np.asarray(H, dtype=np.float64))
        self._transforms[index] = G
        self._order.append(index)
        return G

    def set(self, index: int, G: np.ndarray) -> np.ndarray:
        """Record an explicit mosaic-frame transform for ``index``."""
        G = normalise(G)
        self._transforms[index] = G
        if index not in self._order:
            self._order.append(index)
        return G

    def get(self, index: int) -> Optional[np.ndarray]:
        """Return the mosaic-frame transform of ``index`` if it is known."""
        return self._transforms.get(index)

    def last(self) -> Optional[int]:
        """Return the index of the most recently accumulated frame."""
        return self._order[-1] if self._order else None

    @property
    def order(self) -> List[int]:
        """Indices of the accumulated frames, in insertion order."""
        return list(self._order)


def transformed_corners(H: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return the four image corners mapped through ``H``.

    The corners define the axis aligned bounding box that the warped frame
    occupies on the canvas (paper, Section 3.6).
    """
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def bounding_box(points: np.ndarray) -> tuple:
    """Return the ``(x0, y0, x1, y1)`` bounding box of ``points``."""
    if points.size == 0:
        return 0, 0, 0, 0
    x0 = float(np.min(points[:, 0]))
    y0 = float(np.min(points[:, 1]))
    x1 = float(np.max(points[:, 0]))
    y1 = float(np.max(points[:, 1]))
    return x0, y0, x1, y1
