"""Descriptor matching with Lowe's ratio test.

Following the paper (Section 3.4 and Table 3) tentative correspondences are
obtained with a KNN search in an approximate nearest neighbour structure
(FLANN) for the floating point descriptors, and the ambiguous ones are
suppressed with Lowe's ratio test before any geometric verification happens.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .config import MatcherConfig


class DescriptorMatcher:
    """Thin wrapper around OpenCV's FLANN matcher with the ratio test applied."""

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=config.flann_trees),  # 1 = KD-TREE
            dict(checks=config.flann_checks),
        )

    # ------------------------------------------------------------------- API
    def knn_match(
        self,
        descriptors_a: Optional[np.ndarray],
        descriptors_b: Optional[np.ndarray],
        k: int = 2,
    ) -> List[List[cv2.DMatch]]:
        """Return the ``k`` nearest neighbours of every descriptor of ``a``.

        The ratio test is deliberately *not* applied here: the candidates may
        still have to be filtered, and the ambiguity has to be measured
        inside the set of plausible candidates.
        """
        if descriptors_a is None or descriptors_b is None:
            return []
        if len(descriptors_a) < 2 or len(descriptors_b) < 2:
            return []
        return self._matcher.knnMatch(descriptors_a, descriptors_b, k=k)

    def ratio_filter(
        self, knn: List[List[cv2.DMatch]], ratio: Optional[float] = None
    ) -> List[cv2.DMatch]:
        """Apply Lowe's ratio test to a list of nearest neighbour pairs.

        A match is kept when the best candidate is clearly better than the
        second best one.
        """
        if ratio is None:
            ratio = self.config.ratio
        matches: List[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            best, second = pair[0], pair[1]
            if best.distance < ratio * second.distance:
                matches.append(best)
        return matches
