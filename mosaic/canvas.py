"""Growable mosaic canvas and placement of warped frames.

The canvas holds the orthomosaic under construction:

* ``data``  - the mosaic itself, ``uint8`` BGR.
* ``mask``  - the pixels that actually carry data.
* ``extent``- the logical extent of the mosaic, in mosaic coordinates.

The canvas grows whenever a warped frame reaches outside of the current
extent, which is the behaviour described in Section 3.6 of the paper: the
corners of the incoming frame are transformed with the accumulated transform,
the axis aligned bounding box is computed and the canvas is padded while the
existing mosaic content is preserved.

The allocated buffer is kept slightly larger than the logical extent so that
it does not have to be reallocated on every single step.  The slack lives
*outside* of the logical extent and is therefore never part of the mosaic: the
extent, and not the buffer, defines where the mosaic ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .blending import blend
from .config import BlendingConfig, CanvasConfig
from .transforms import bounding_box, transformed_corners


@dataclass
class PlacementResult:
    """Statistics of a single frame placement."""

    #: Bounding box of the warped frame in mosaic coordinates.
    bbox: Tuple[float, float, float, float]
    #: Region of interest that was actually processed, in canvas pixels.
    roi: Tuple[int, int, int, int]
    #: Number of pixels of the mosaic that were modified.
    written: int
    #: Number of pixels covered by the warped frame inside the canvas.
    covered: int
    #: Mean absolute difference between the warped frame and the content it
    #: covers, in the overlap region.  It measures how well the frame is
    #: aligned with the mosaic: a misregistered frame leaves a visible seam
    #: and a high value.
    misalignment: float = 0.0
    #: True when the frame did not add any new coverage.
    empty: bool = False


class MosaicCanvas:
    """A mosaic canvas that can grow while frames are added to it."""

    def __init__(
        self,
        canvas_config: CanvasConfig,
        blending_config: BlendingConfig,
    ) -> None:
        self.config = canvas_config
        self.blending_config = blending_config
        self.data: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        #: Mosaic coordinates of the pixel ``(0, 0)`` of the buffer.
        self._buffer_origin: Tuple[float, float] = (0.0, 0.0)
        #: Logical extent of the mosaic inside the buffer, in buffer pixels.
        self._extent: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # ------------------------------------------------------------------ shape
    @property
    def scale(self) -> float:
        return self.config.scale

    @property
    def extent(self) -> Tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` of the logical region in buffer pixels."""
        return self._extent

    @property
    def origin(self) -> Tuple[float, float]:
        """Return the mosaic coordinates of the top left corner of the mosaic."""
        bx, by = self._buffer_origin
        x0, y0, _, _ = self._extent
        return (bx + x0 / self.scale, by + y0 / self.scale)

    @property
    def size(self) -> Tuple[int, int]:
        """Return ``(width, height)`` of the mosaic in pixels."""
        x0, y0, x1, y1 = self._extent
        return (x1 - x0, y1 - y0)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return ``(height, width)`` of the mosaic in pixels."""
        width, height = self.size
        return (height, width)

    @property
    def is_empty(self) -> bool:
        return self.data is None

    @property
    def footprint_mb(self) -> float:
        """Memory used by the mosaic (colour + validity mask), in megabytes."""
        width, height = self.size
        return width * height * 4 / (1024 * 1024)

    # ------------------------------------------------------------- life cycle
    def initialize(self, image: np.ndarray) -> None:
        """Seed the canvas with the first frame of the sequence."""
        h, w = image.shape[:2]
        scale = self.scale
        if scale != 1.0:
            size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        margin = max(0, int(self.config.grow_margin))
        self.data = np.zeros((h + margin, w + margin, 3), np.uint8)
        self.mask = np.zeros((h + margin, w + margin), np.uint8)
        self.data[:h, :w] = image
        self.mask[:h, :w] = 255
        self._buffer_origin = (0.0, 0.0)
        self._extent = (0, 0, w, h)

    # ---------------------------------------------------------------- helpers
    def _to_buffer(self, x: float, y: float) -> Tuple[float, float]:
        """Convert mosaic coordinates into buffer pixels."""
        bx, by = self._buffer_origin
        return ((x - bx) * self.scale, (y - by) * self.scale)

    def _grow(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Extend the logical extent so that it covers the given coordinates.

        The buffer is only reallocated when the new extent does not fit into
        it any more; the slack that is added then lives outside of the logical
        extent and never becomes part of the mosaic.
        """
        assert self.data is not None and self.mask is not None
        ex0, ey0, ex1, ey1 = self._extent

        # Required extent, expressed in buffer pixels.
        rx0, ry0 = self._to_buffer(x0, y0)
        rx1, ry1 = self._to_buffer(x1, y1)
        need = (
            int(np.floor(min(ex0, rx0))),
            int(np.floor(min(ey0, ry0))),
            int(np.ceil(max(ex1, rx1))),
            int(np.ceil(max(ey1, ry1))),
        )

        if need == self._extent:
            return

        buf_h, buf_w = self.data.shape[:2]
        if need[2] <= buf_w and need[3] <= buf_h and need[0] >= 0 and need[1] >= 0:
            # The new extent still fits into the buffer.
            self._extent = need
            return

        # The buffer has to be reallocated.  It is enlarged with a margin on
        # every side so that the next reallocation happens a long time from
        # now.
        margin = max(1, int(self.config.grow_margin))
        new_w = max(buf_w, need[2] - need[0]) + margin
        new_h = max(buf_h, need[3] - need[1]) + margin
        new_x0 = need[0] - margin
        new_y0 = need[1] - margin

        if self.config.max_pixels and new_w * new_h > self.config.max_pixels:
            raise MemoryError(
                f"the mosaic canvas would reach {new_w}x{new_h} pixels "
                f"({new_w * new_h / 1e6:.0f} Mpx, about "
                f"{new_w * new_h * 4 / 1e9:.1f} GB for the colour and the mask), "
                f"which exceeds the configured limit of "
                f"{self.config.max_pixels / 1e6:.0f} Mpx. Lower --canvas-scale "
                f"or raise --max-canvas-pixels."
            )

        data = np.zeros((new_h, new_w, 3), np.uint8)
        mask = np.zeros((new_h, new_w), np.uint8)
        dx = ex0 - new_x0
        dy = ey0 - new_y0
        data[dy : dy + (ey1 - ey0), dx : dx + (ex1 - ex0)] = self.data[
            ey0:ey1, ex0:ex1
        ]
        mask[dy : dy + (ey1 - ey0), dx : dx + (ex1 - ex0)] = self.mask[
            ey0:ey1, ex0:ex1
        ]

        self.data = data
        self.mask = mask
        self._buffer_origin = (
            self._buffer_origin[0] + new_x0 / self.scale,
            self._buffer_origin[1] + new_y0 / self.scale,
        )
        self._extent = (
            need[0] - new_x0,
            need[1] - new_y0,
            need[2] - new_x0,
            need[3] - new_y0,
        )

    # -------------------------------------------------------------------- API
    def place(self, image: np.ndarray, G: np.ndarray) -> PlacementResult:
        """Warp ``image`` onto the canvas and blend it with the mosaic.

        ``G`` maps the pixels of ``image`` into the mosaic coordinate system.
        """
        if self.data is None:
            raise RuntimeError("the canvas has not been initialised")

        h, w = image.shape[:2]
        corners = transformed_corners(G, w, h)
        bbox = bounding_box(corners)
        self._grow(bbox[0], bbox[1], bbox[2], bbox[3])

        assert self.data is not None and self.mask is not None
        ex0, ey0, ex1, ey1 = self._extent
        scale = self.scale

        # When the mosaic is built at a lower resolution than the source, the
        # frame is resampled first: an area filter keeps the detail of the
        # original instead of aliasing it, and the warp then works on a much
        # smaller image.
        if scale != 1.0:
            small = cv2.resize(
                image,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            h, w = small.shape[:2]
            G = np.asarray(G, dtype=np.float64) @ np.diag(
                [1.0 / scale, 1.0 / scale, 1.0]
            )
            image = small

        # Region of interest of the warped frame, in buffer pixels.
        fx0, fy0 = self._to_buffer(bbox[0], bbox[1])
        fx1, fy1 = self._to_buffer(bbox[2], bbox[3])
        rx0 = int(np.floor(fx0))
        ry0 = int(np.floor(fy0))
        rx1 = int(np.ceil(fx1))
        ry1 = int(np.ceil(fy1))
        # The blend only touches the part of the frame that is inside the
        # mosaic; the rest of the buffer stays untouched.
        rx0 = max(rx0, ex0)
        ry0 = max(ry0, ey0)
        rx1 = min(rx1, ex1)
        ry1 = min(ry1, ey1)
        roi_w, roi_h = rx1 - rx0, ry1 - ry0
        if roi_w <= 0 or roi_h <= 0:
            return PlacementResult(bbox, (rx0, ry0, rx1, ry1), 0, 0, True)

        # Mosaic -> buffer -> ROI local transform.
        bx, by = self._buffer_origin
        to_buffer = np.array(
            [[scale, 0.0, -bx * scale], [0.0, scale, -by * scale], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        to_roi = np.array(
            [[1.0, 0.0, -rx0], [0.0, 1.0, -ry0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        M = to_roi @ to_buffer @ np.asarray(G, dtype=np.float64)

        warped = cv2.warpPerspective(
            image,
            M,
            (roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        source_mask = np.full((h, w), 255, np.uint8)
        valid = cv2.warpPerspective(
            source_mask,
            M,
            (roi_w, roi_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        base_image = self.data[ry0:ry1, rx0:rx1]
        base_mask = self.mask[ry0:ry1, rx0:rx1]

        # How well does the frame agree with the content it covers?  This is
        # measured before anything is written, so it is a direct measure of
        # the alignment of the frame with the mosaic.
        overlap = (valid > 0) & (base_mask > 0)
        misalignment = 0.0
        if overlap.any():
            difference = np.abs(
                warped.astype(np.int16) - base_image.astype(np.int16)
            ).mean(axis=2)
            misalignment = float(difference[overlap].mean())

        blended, apply_mask = blend(
            base_image, base_mask, warped, valid, self.blending_config
        )

        written = int(np.count_nonzero(apply_mask))
        if written:
            region = apply_mask > 0
            base_image[region] = blended[region]
            self.data[ry0:ry1, rx0:rx1] = base_image

        covered = int(np.count_nonzero(valid))
        if covered:
            self.mask[ry0:ry1, rx0:rx1] = np.where(
                valid > 0, 255, self.mask[ry0:ry1, rx0:rx1]
            ).astype(np.uint8)

        return PlacementResult(
            bbox=bbox,
            roi=(rx0, ry0, rx1, ry1),
            written=written,
            covered=covered,
            misalignment=misalignment,
            empty=covered == 0,
        )

    # ------------------------------------------------------------------- I/O
    def mask_snapshot(self, copy: bool = True) -> np.ndarray:
        """Return the validity mask of the mosaic (255 where it has data)."""
        if self.mask is None:
            raise RuntimeError("the canvas has not been initialised")
        x0, y0, x1, y1 = self._extent
        view = self.mask[y0:y1, x0:x1]
        return view.copy() if copy else view

    def snapshot(self, copy: bool = True) -> np.ndarray:
        """Return the current mosaic.

        ``copy=False`` returns a view on the live canvas, which is enough for
        read only consumers such as the JPEG writer and avoids copying a
        canvas that can be several hundred megapixels.
        """
        if self.data is None:
            raise RuntimeError("the canvas has not been initialised")
        x0, y0, x1, y1 = self._extent
        view = self.data[y0:y1, x0:x1]
        return view.copy() if copy else view
