"""Incremental feature store of the mosaic.

The pipeline of the paper registers every frame against its **predecessor**
and accumulates the inter-frame transforms (Eq. 6).  The transform of frame
``k`` therefore contains the sum of all the errors made on the way, which
shows up as a growing misalignment - and as visible seams - over a long
sequence.  The paper acknowledges this as a limitation ("drift may accumulate
as sequences length increases").

This module holds the SIFT features of the mosaic, expressed in mosaic
coordinates, so that an incoming frame can be registered against the mosaic
and not only against its predecessor.  The correspondences it provides come in
two flavours, and they are used for two different things:

* the ones with the frames captured just before it describe the **relative
  motion** of the incoming frame.  They are numerous and reliable, but they
  only ever reproduce the drift that is already there;
* the ones with the frames captured further back describe where the frame
  **sits over the whole mosaic**.  There are fewer of them and they are
  noisier, but they are the only evidence that ties the part of the flight
  the frame belongs to to the rest of the mosaic.

Two properties keep the store small and the matching well behaved:

* one descriptor per ground location - a new observation of a location
  *replaces* the previous one.  Duplicated descriptors would break Lowe's
  ratio test, because the second best candidate of a match would be a copy of
  the best one;
* the store only holds the keypoints and the descriptors of the frames, not
  any pixel data, so its memory stays in the order of a few megabytes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import MosaicStoreConfig
from .features import FeatureSet


class MosaicFeatureStore:
    """SIFT features of the mosaic, in mosaic coordinates."""

    def __init__(self, config: MosaicStoreConfig) -> None:
        self.config = config
        self._points = np.empty((0, 2), np.float32)
        self._descriptors: Optional[np.ndarray] = None
        self._frames = np.empty((0,), np.int32)
        self._grid: Dict[Tuple[int, int], List[int]] = {}
        self._next_frame_id = 0

    # ------------------------------------------------------------------ state
    def __len__(self) -> int:
        return int(self._points.shape[0])

    @property
    def points(self) -> np.ndarray:
        """Keypoints of the store, in mosaic coordinates."""
        return self._points

    @property
    def descriptors(self) -> Optional[np.ndarray]:
        """Descriptors of the store."""
        return self._descriptors

    def as_featureset(self) -> FeatureSet:
        """Return the content of the store as a :class:`FeatureSet`."""
        return FeatureSet(
            points=self._points,
            descriptors=self._descriptors,
            frame_ids=self._frames,
        )

    def recent(self, window: int) -> FeatureSet:
        """Return the features of the last ``window`` frames.

        Those are the ones the incoming frame actually overlaps, so they are
        the only candidates that are both numerous and spatially constrained.
        """
        if window <= 0 or self._descriptors is None or len(self._points) == 0:
            return self.as_featureset()

        newest = int(self._frames.max())
        keep = self._frames > newest - window
        if not keep.any():
            return self.as_featureset()
        return FeatureSet(
            points=self._points[keep],
            descriptors=self._descriptors[keep],
            frame_ids=self._frames[keep],
        )

    def query(self, x0: float, y0: float, x1: float, y1: float) -> FeatureSet:
        """Return the features inside a rectangle of the mosaic.

        The rectangle is given in mosaic coordinates.  This is the *spatial*
        neighbourhood of a frame: a flight that comes back over ground it
        covered earlier puts the frame close to frames that were captured a
        long time ago, and those are the candidates that tie it to the rest of
        the mosaic.
        """
        if self._descriptors is None or len(self._points) == 0:
            return FeatureSet(points=np.empty((0, 2), np.float32), descriptors=None)

        cell = self._cell()
        cx0 = int(np.floor(x0 / cell))
        cy0 = int(np.floor(y0 / cell))
        cx1 = int(np.floor(x1 / cell))
        cy1 = int(np.floor(y1 / cell))

        indices: List[int] = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                bucket = self._grid.get((cx, cy))
                if bucket:
                    indices.extend(bucket)

        if not indices:
            return FeatureSet(points=np.empty((0, 2), np.float32), descriptors=None)

        index = np.asarray(sorted(indices), dtype=np.int64)
        return FeatureSet(
            points=self._points[index],
            descriptors=self._descriptors[index],
            frame_ids=self._frames[index],
        )

    # ------------------------------------------------------------------ update
    def update(
        self, features: FeatureSet, G: np.ndarray, frame_id: Optional[int] = None
    ) -> int:
        """Add the features of a frame to the store.

        The keypoints are mapped into the mosaic coordinates with ``G`` and
        replace the stored observations that are closer than
        ``config.min_distance``.  Returns the number of stored features.
        """
        if features is None or len(features) == 0 or features.descriptors is None:
            return len(self)

        warped = cv2.perspectiveTransform(
            features.points.reshape(-1, 1, 2).astype(np.float64),
            np.asarray(G, dtype=np.float64),
        ).reshape(-1, 2)

        replaced = self._replaced_indices(warped)

        if frame_id is None:
            fid = self._next_frame_id
            self._next_frame_id += 1
        else:
            fid = int(frame_id)
            self._next_frame_id = max(self._next_frame_id, fid + 1)

        if self._descriptors is None or len(self._points) == 0:
            points = warped.astype(np.float32)
            descriptors = features.descriptors
            frames = np.full(len(warped), fid, np.int32)
        else:
            keep = np.ones(len(self._points), bool)
            if len(replaced):
                keep[replaced] = False
            points = np.vstack([self._points[keep], warped.astype(np.float32)])
            descriptors = np.vstack([self._descriptors[keep], features.descriptors])
            frames = np.concatenate(
                [self._frames[keep], np.full(len(warped), fid, np.int32)]
            )

        self._points = points
        self._descriptors = descriptors
        self._frames = frames

        self._enforce_limit()
        self._rebuild_grid()
        return len(self)

    # ---------------------------------------------------------------- private
    def _cell(self) -> float:
        # The cell has to be large enough for the 3x3 neighbourhood search of
        # the replacement logic to cover the minimum distance.
        return max(64.0, float(self.config.min_distance))

    def _replaced_indices(self, new_points: np.ndarray) -> np.ndarray:
        """Return the indices of the stored features that are re-observed."""
        if len(self._points) == 0:
            return np.empty((0,), np.int64)

        cell = self._cell()
        # NOTE: the search cell (>= 64 px) is used as the replacement radius.
        # Changing this alters the feature density and therefore the produced
        # mosaics; it is intentionally left as-is for output compatibility.
        cell_sq = cell * cell
        replaced = set()

        for point in new_points:
            cx = int(np.floor(point[0] / cell))
            cy = int(np.floor(point[1] / cell))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bucket = self._grid.get((cx + dx, cy + dy))
                    if not bucket:
                        continue
                    indices = np.asarray(bucket, dtype=np.int64)
                    # vectorised distance test over the whole bucket
                    diffs = self._points[indices] - point
                    dists_sq = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
                    close = indices[dists_sq < cell_sq]
                    for idx in close:
                        replaced.add(int(idx))
        return np.fromiter(replaced, dtype=np.int64, count=len(replaced))

    def _rebuild_grid(self) -> None:
        cell = self._cell()
        grid: Dict[Tuple[int, int], List[int]] = {}
        for index, point in enumerate(self._points):
            key = (int(np.floor(point[0] / cell)), int(np.floor(point[1] / cell)))
            grid.setdefault(key, []).append(index)
        self._grid = grid

    def _enforce_limit(self) -> None:
        """Drop the oldest frame when the store grows past its limit."""
        limit = int(self.config.max_features)
        if limit <= 0:
            return
        while len(self._points) > limit:
            oldest = int(self._frames.min())
            keep = self._frames != oldest
            self._points = self._points[keep]
            self._descriptors = self._descriptors[keep]
            self._frames = self._frames[keep]
