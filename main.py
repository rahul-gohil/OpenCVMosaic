#!/usr/bin/env python3
"""Incremental 2D orthomosaic construction from an ordered image sequence.

The pipeline follows the processing procedure of the paper "Real-Time 2D
Orthomosaic Mapping from UAV Video via Feature-Based Image Registration"
(Appl. Sci. 2026, 16, 2133), restricted to still images:

    read -> SIFT -> match -> homography -> warp onto the canvas -> blend
    (overwrite or feather) -> write the canvas

Example
-------
    python main.py --dataset datasets/UAVPistachio/images_nadir_RGB \
                   --output-root outputs --run-name pistachio

Every update of the canvas is written twice into ``<output-root>/<run-name>``:
once at the native resolution of the mosaic and once scaled down by 10%.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from typing import Optional, Sequence

from mosaic.config import (
    BlendingConfig,
    CanvasConfig,
    DetectorConfig,
    MatcherConfig,
    MosaicConfig,
    MosaicStoreConfig,
    OutputConfig,
    RegistrationConfig,
)
from mosaic.pipeline import StitchingPipeline
from mosaic.merge import MosaicMerger
from mosaic.chunker import ChunkRunner

DEFAULT_DATASET = "datasets/UAVPistachio/images_nadir_RGB"


def config_from_args(args: argparse.Namespace) -> MosaicConfig:
    """Build the pipeline configuration from the parsed arguments."""
    if args.canvas_scale <= 0:
        raise ValueError("--canvas-scale must be > 0")
    detector = DetectorConfig()
    matcher = MatcherConfig()
    registration = RegistrationConfig(
        mode=args.registration,
        grounding_window=args.grounding_window,
        min_inliers=args.min_inliers,
    )
    store = MosaicStoreConfig(
        min_distance=args.store_min_distance,
        query_margin=args.query_margin,
        max_features=args.store_max_features,
    )
    blending = BlendingConfig(method=args.blend, ramp=args.blend_ramp)
    canvas = CanvasConfig(scale=args.canvas_scale, max_pixels=args.max_canvas_pixels)
    output = OutputConfig(
        root=args.output_root,
        run_name=args.run_name,
        save_preview=not args.no_preview,
        preview_scale=args.preview_scale,
        save_timings=not args.no_timings,
        save_debug=args.save_debug,
        jpeg_quality=args.jpeg_quality,
    )
    return MosaicConfig(
        dataset_dir=args.dataset,
        image_pattern=args.pattern,
        start=args.start,
        end=args.end,
        relink_on_failure=not args.no_relink,
        detector=detector,
        matcher=matcher,
        registration=registration,
        store=store,
        blending=blending,
        canvas=canvas,
        output=output,
    )


def _run_pipeline(args: argparse.Namespace) -> int:
    """Run the per-frame pipeline."""
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start = time.perf_counter()
    try:
        config = config_from_args(args)
        pipeline = StitchingPipeline(config, run_name=run_name)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    pipeline.run()
    elapsed = time.perf_counter() - start
    print(f" wall clock: {elapsed:.3f} s")
    return 0


def _run_merge(args: argparse.Namespace) -> int:
    """Run the chunk merge."""
    if args.detect_px <= 0:
        print("error: --detect-px must be > 0", file=sys.stderr)
        return 2
    if args.features < 4:
        print("error: --features must be >= 4", file=sys.stderr)
        return 2
    merger = MosaicMerger(
        inputs=args.inputs,
        out_dir=args.out,
        features=args.features,
        detect_px=args.detect_px,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        threshold=args.threshold,
        quality=args.quality,
    )
    start = time.perf_counter()
    try:
        result = merger.run()
    except RuntimeError as error:
        print(f"merge failed: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start
    if result is None:
        print("merge failed", file=sys.stderr)
        return 1
    print(f" wall clock: {elapsed:.3f} s")
    return 0


def _run_chunks(args: argparse.Namespace) -> int:
    """Run a dataset in overlapping chunks."""
    runner = ChunkRunner(
        dataset=args.dataset,
        chunk_size=args.chunk_size,
        output_root=args.output_root,
        prefix=args.prefix,
        pattern=args.pattern,
        extra_args=args.extra or [],
    )
    start = time.perf_counter()
    try:
        runner.run()
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"chunked run failed: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start
    print(f" wall clock: {elapsed:.3f} s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser of the pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a 2D orthomosaic from an ordered image sequence with SIFT "
            "matching, a planar homography and the blending rule of the paper."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="subcommand")

    # --- run (default pipeline) ---
    run_parser = subparsers.add_parser("run", help="Run the per-frame pipeline on a dataset")
    _add_run_args(run_parser)

    # --- merge ---
    merge_parser = subparsers.add_parser("merge", help="Merge chunked mosaics into one")
    merge_parser.add_argument("--inputs", nargs="+", required=True,
                            help="the mosaics to merge, in order (glob patterns allowed)")
    merge_parser.add_argument("--out", default="outputs/merged", help="output folder")
    merge_parser.add_argument("--features", type=int, default=4000,
                            help="keypoints detected on each mosaic")
    merge_parser.add_argument("--detect-px", type=float, default=4e6,
                            help="pixel budget of the downscaled copy used for detection")
    merge_parser.add_argument("--min-inliers", type=int, default=8,
                            help="minimum inliers to accept a merge")
    merge_parser.add_argument("--min-inlier-ratio", type=float, default=0.03,
                            help="minimum inlier ratio to accept a merge")
    merge_parser.add_argument("--threshold", type=int, default=10,
                            help="grey level above which a pixel counts as content")
    merge_parser.add_argument("--quality", type=int, default=92, help="JPEG quality")

    # --- run-chunks ---
    chunks_parser = subparsers.add_parser("run-chunks", help="Run a dataset in overlapping chunks")
    chunks_parser.add_argument("--dataset", default=DEFAULT_DATASET, help="folder with the images")
    chunks_parser.add_argument("--pattern", default="*.JPG", help="glob pattern of the images")
    chunks_parser.add_argument("--chunk-size", type=int, default=17, help="frames per chunk")
    chunks_parser.add_argument("--output-root", default="outputs", help="root folder for chunk outputs")
    chunks_parser.add_argument("--prefix", default=None, help="prefix for chunk folder names (default: dataset basename)")
    chunks_parser.add_argument("--extra", nargs="*", default=None, help="extra args passed to run")

    # --- backward compat: top-level args (no subcommand) ---
    # If no subcommand is given, treat as 'run' for backward compatibility
    _add_run_args(parser)

    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add the run/pipeline arguments to a parser."""
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="folder with the images")
    parser.add_argument("--pattern", default="*.JPG", help="glob pattern of the images")
    parser.add_argument("--start", type=int, default=None, help="index of the first frame")
    parser.add_argument("--end", type=int, default=None, help="index of the last frame")

    group = parser.add_argument_group("registration")
    group.add_argument(
        "--registration",
        default="mosaic",
        choices=["mosaic", "frame"],
        help=(
            "register a frame against the accumulated mosaic (no drift) or "
            "against its predecessor and accumulate the transforms"
        ),
    )
    group.add_argument(
        "--min-inliers", type=int, default=20, help="minimum inlier support to accept a frame"
    )
    group.add_argument(
        "--store-min-distance",
        type=float,
        default=40.0,
        help="minimum distance in pixels between two features of the mosaic store",
    )
    group.add_argument(
        "--query-margin",
        type=float,
        default=1500.0,
        help="margin around the predicted position used to select the store features",
    )
    group.add_argument(
        "--grounding-window",
        type=int,
        default=6,
        help="number of recent frames whose matches describe the relative motion",
    )
    group.add_argument(
        "--store-max-features",
        type=int,
        default=50000,
        help="maximum number of features kept in the mosaic store",
    )
    group.add_argument(
        "--no-relink",
        action="store_true",
        help="drop a frame instead of re-linking it to the last accepted one",
    )

    group = parser.add_argument_group("blending")
    group.add_argument(
        "--blend",
        default="overwrite",
        choices=["feather", "overwrite"],
        help="blending strategy (overwrite = the method of the paper)",
    )
    group.add_argument(
        "--blend-ramp", type=int, default=32, help="width of the transition band in pixels"
    )

    group = parser.add_argument_group("canvas")
    group.add_argument(
        "--canvas-scale",
        type=float,
        default=1.0,
        help="resolution of the mosaic relative to the source images",
    )
    group.add_argument(
        "--max-canvas-pixels",
        type=int,
        default=0,
        help="abort if the canvas grows beyond this many pixels (0 = no limit)",
    )

    group = parser.add_argument_group("output")
    group.add_argument("--output-root", default="outputs", help="root folder of the runs")
    group.add_argument("--run-name", default=None, help="name of the run folder")
    group.add_argument(
        "--no-preview", action="store_true", help="do not write the downscaled canvas"
    )
    group.add_argument(
        "--preview-scale", type=float, default=0.1, help="scale of the downscaled canvas"
    )
    group.add_argument("--jpeg-quality", type=int, default=90, help="quality of the JPEGs")
    group.add_argument("--no-timings", action="store_true", help="do not write the timings CSV")
    group.add_argument(
        "--save-debug", action="store_true", help="write match visualisations"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "merge":
        return _run_merge(args)
    elif args.command == "run-chunks":
        return _run_chunks(args)
    else:
        # Default: run the per-frame pipeline (backward compatible)
        return _run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
