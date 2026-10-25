"""Ordered discovery and loading of the source image sequence.

The dataset is expected to expose a flat directory of still images whose file
names carry a sequence number (``DJI_0002.JPG``, ``DJI_0003.JPG``, ...).  The
images are ordered by that number so that the sequence follows the acquisition
order of the flight.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

_NUMBER_RE = re.compile(r"(\d+)")


def natural_key(path: Path) -> Tuple:
    """Return a sort key that orders file names by their embedded numbers.

    ``DJI_0002.JPG`` sorts before ``DJI_0010.JPG`` because the numeric group is
    compared as an integer rather than as a string.
    """
    parts = _NUMBER_RE.split(path.name)
    key: List = [int(p) if p.isdigit() else p.lower() for p in parts]
    return tuple(key)


def find_images(directory: str, pattern: str) -> List[Path]:
    """Return the images of ``directory`` matching ``pattern``.

    Matching is case-insensitive: a ``*.JPG`` pattern also finds ``*.jpg``
    files (and vice versa), which matters on case-sensitive filesystems.
    """
    base = Path(directory)
    if not base.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {base}")
    variants = {pattern, pattern.lower(), pattern.upper()}
    found = set()
    for variant in variants:
        found.update(base.glob(variant))
    return sorted(found, key=natural_key)


class ImageSequence:
    """An ordered, immutable view over a directory of images.

    Parameters
    ----------
    directory:
        Folder holding the frames.
    pattern:
        Glob pattern used to select the frames.
    start, end:
        Optional inclusive index bounds applied *after* ordering.
    """

    def __init__(
        self,
        directory: str,
        pattern: str = "*.JPG",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        self.directory = Path(directory)
        found = find_images(directory, pattern)
        if not found:
            raise FileNotFoundError(
                f"no image matching '{pattern}' found in {self.directory}"
            )

        lo = 0 if start is None else max(0, start)
        hi = len(found) if end is None else min(len(found), end + 1)
        if lo >= hi:
            raise ValueError(f"empty index range [{lo}, {hi}) for {self.directory}")
        self.paths: List[Path] = found[lo:hi]

    # ------------------------------------------------------------------ dunder
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Path:
        return self.paths[index]

    # ------------------------------------------------------------------- API
    def name(self, index: int) -> str:
        """Return the file name of the frame at ``index``."""
        return self.paths[index].name

    def load(self, index: int) -> np.ndarray:
        """Read the frame at ``index`` as an 8-bit BGR image."""
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"failed to read image: {self.paths[index]}")
        return image

    def names(self) -> Sequence[str]:
        """Return the ordered file names of the sequence."""
        return [p.name for p in self.paths]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ImageSequence({self.directory}, {len(self.paths)} frames)"
