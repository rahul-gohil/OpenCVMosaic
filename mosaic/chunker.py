"""Run a dataset in overlapping chunks.

Each chunk is an independent run of the pipeline with its own canvas and its
own output folder.  Consecutive chunks share one frame so that they can be
merged afterwards.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, List, Optional


class ChunkRunner:
    """Run a dataset in overlapping chunks.

    Parameters
    ----------
    dataset : str
        Folder holding the images.
    chunk_size : int
        Number of frames per chunk (consecutive chunks share the last frame).
    output_root : str
        Root folder for chunk outputs.
    prefix : str
        Prefix for chunk folder names.
    pattern : str
        Glob pattern for selecting images.
    extra_args : list of str
        Extra CLI arguments passed to ``main.py``.
    python : str
        Path to the Python interpreter.
    log : callable
        Logging function.
    """

    def __init__(
        self,
        dataset: str,
        chunk_size: int = 17,
        output_root: str = "outputs",
        prefix: Optional[str] = None,
        pattern: str = "*.JPG",
        extra_args: Optional[List[str]] = None,
        python: str = sys.executable,
        log: Callable[[str], None] = print,
    ) -> None:
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1")
        self.dataset = dataset
        self.chunk_size = int(chunk_size)
        self.output_root = output_root
        if prefix is None:
            self.prefix = os.path.basename(os.path.normpath(dataset))
        else:
            self.prefix = prefix
        self.pattern = pattern
        self.extra_args = extra_args or []
        self.python = python
        self.log = log

    def _count_frames(self) -> int:
        """Count the number of frames matching the pattern."""
        from .dataset import find_images
        return len(find_images(self.dataset, self.pattern))

    def run(self) -> List[str]:
        """Run all chunks and return the list of output folder names."""
        total = self._count_frames()
        self.log(f"dataset: {self.dataset}   frames: {total}   chunk: {self.chunk_size}")

        start = 0
        n = 1
        chunk_names = []
        while start < total:
            end = start + self.chunk_size - 1
            if end >= total:
                end = total - 1
            name = f"{self.prefix}_{start + 1:03d}_{end + 1:03d}"
            self.log(f"=== chunk {n} : frames {start + 1}-{end + 1}  -> {self.output_root}/{name}")

            main_py = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"
            )
            cmd = [
                self.python, "-u", main_py,
                "--dataset", self.dataset,
                "--pattern", self.pattern,
                "--start", str(start),
                "--end", str(end),
                "--output-root", self.output_root,
                "--run-name", name,
            ] + self.extra_args

            proc = subprocess.run(cmd, capture_output=True, text=True)
            # Print summary lines
            for line in proc.stdout.splitlines():
                if any(k in line for k in ("run summary", "frames processed", "accepted",
                                           "rejected", "final canvas", "total time",
                                           "mean per step", "wall clock")):
                    self.log(line)
            if proc.returncode != 0:
                detail = proc.stderr[-500:] if proc.stderr else "no stderr"
                self.log(f"ERROR in chunk {n}:")
                self.log(detail)
                raise RuntimeError(f"chunk {name} failed (exit {proc.returncode}): {detail}")

            chunk_names.append(name)

            # Stop if we've reached the last frame
            if end >= total - 1:
                break
            start = end
            n += 1

        self.log(f"=== {n} chunks done")
        return chunk_names