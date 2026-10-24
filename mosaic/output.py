"""Artefact writing for a mosaic run.

Every update of the canvas is written to the run folder twice: once at the
native resolution of the mosaic and once downscaled (10% by default) so that
the intermediate state can be inspected quickly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import OutputConfig


def resize(image: np.ndarray, ratio: float) -> np.ndarray:
    """Downscale ``image`` by ``ratio`` using an area filter."""
    if ratio == 1.0:
        return image
    h, w = image.shape[:2]
    size = (max(1, int(round(w * ratio))), max(1, int(round(h * ratio))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class RunOutput:
    """Collects the artefacts of a single run inside its own folder."""

    def __init__(self, config: OutputConfig, run_name: str) -> None:
        self.config = config
        self.run_name = run_name
        self.directory = Path(config.root) / run_name
        self.directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _canvas_path(self, index: int, suffix: str) -> Path:
        return self.directory / f"canvas_{index:04d}{suffix}.jpg"

    # ------------------------------------------------------------------- API
    def write_preview(self, canvas: np.ndarray, index: int) -> Optional[Path]:
        """Write the downscaled preview of the current mosaic."""
        if not self.config.save_preview:
            return None
        path = self._canvas_path(index, f"_x{self.config.preview_scale:g}")
        preview = resize(canvas, self.config.preview_scale)
        if cv2.imwrite(
            str(path), preview, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        ):
            return path
        return None

    def write_canvas(self, canvas: np.ndarray, index: int) -> Dict[str, Optional[Path]]:
        """Write the artefacts of the current state of the mosaic.

        The downscaled preview is written after every update.  The mosaic at
        its native resolution is written once, when the run terminates, because
        it becomes large quickly and the encoding would dominate the runtime
        of the pipeline.
        """
        written: Dict[str, Optional[Path]] = {"full": None, "preview": None}
        written["preview"] = self.write_preview(canvas, index)
        return written

    def write_debug(self, image: np.ndarray, name: str) -> Optional[Path]:
        """Write a diagnostic image (match visualisations, ...)."""
        if not self.config.save_debug:
            return None
        path = self.directory / f"{name}.jpg"
        if cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]):
            return path
        return None

    def write_final(self, canvas: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[Path]:
        """Write the final mosaic at its native resolution.

        The validity mask is written next to it: the mosaic only covers the
        parts of the scene that were actually imaged, and the empty part of
        the canvas has to be excluded when the mosaic is combined with
        another one.
        """
        if not self.config.save_final:
            return None
        path = self.directory / "mosaic_final.jpg"
        written = None
        if cv2.imwrite(
            str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        ):
            written = path
        if mask is not None:
            mask_path = self.directory / "mosaic_final_mask.png"
            cv2.imwrite(str(mask_path), mask)
        return written

    def write_timings(self, rows: List[Dict[str, Any]]) -> Optional[Path]:
        """Write the per step timings as a CSV file."""
        if not self.config.save_timings or not rows:
            return None
        path = self.directory / "timings.csv"
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_report(self, report: Dict[str, Any]) -> Optional[Path]:
        """Write the run report as JSON."""
        path = self.directory / "report.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        return path

    def write_config(self, config: Dict[str, Any]) -> Optional[Path]:
        """Write the configuration of the run as JSON."""
        path = self.directory / "config.json"
        path.write_text(json.dumps(config, indent=2, default=str))
        return path
