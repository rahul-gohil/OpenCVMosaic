#!/usr/bin/env python3
"""
Pipeline Runner - Runs on EC2 instances (Graviton5 with COOL AMI).
Executes the orthomosaic pipeline (stitch + merge) and reports timings.

This replaces the MCP server (opencv-mcp-server.py) which only ran
findContours on individual images.  Instead, it runs the full pipeline
on a dataset and returns per-stage metrics.

Usage (on EC2):
    python3 pipeline-runner.py --s3-input s3://bucket/inputs/pistachio \
                                --s3-output s3://bucket/outputs/run_001 \
                                --chunk-size 18 \
                                --work-dir /tmp/scratchpad
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path


def run_pipeline(s3_input: str, s3_output: str, chunk_size: int, work_dir: str) -> dict:
    """Run the full orthomosaic pipeline and return metrics."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    # Step 1: Download dataset from S3
    input_dir = work / "input"
    input_dir.mkdir(exist_ok=True)
    dl_start = time.perf_counter()
    subprocess.run(
        ["aws", "s3", "sync", s3_input, str(input_dir)],
        check=True, capture_output=True,
    )
    dl_time = time.perf_counter() - dl_start

    # Step 2: Run chunked stitching
    stitch_start = time.perf_counter()
    subprocess.run(
        [sys.executable, "main.py", "run-chunks",
         "--dataset", str(input_dir),
         "--chunk-size", str(chunk_size),
         "--output-root", str(work / "chunks"),
         "--prefix", "chunk"],
        check=True, cwd="/opt/OpenCVMosaic",
    )
    stitch_time = time.perf_counter() - stitch_start

    # Step 3: Run merge
    merge_start = time.perf_counter()
    subprocess.run(
        [sys.executable, "main.py", "merge",
         "--inputs", str(work / "chunks" / "*" / "mosaic_final.jpg"),
         "--out", str(work / "merged")],
        check=True, cwd="/opt/OpenCVMosaic",
    )
    merge_time = time.perf_counter() - merge_start

    # Step 4: Upload results to S3
    ul_start = time.perf_counter()
    subprocess.run(
        ["aws", "s3", "sync", str(work / "merged"), s3_output],
        check=True, capture_output=True,
    )
    ul_time = time.perf_counter() - ul_start

    total_time = time.perf_counter() - start

    # Collect metrics
    import os as _os
    merged_dir = work / "merged"
    final_mosaic = merged_dir / "mosaic_merged.jpg"
    final_mask = merged_dir / "mosaic_merged_mask.png"

    metrics = {
        "total_time_s": round(total_time, 2),
        "download_time_s": round(dl_time, 2),
        "stitch_time_s": round(stitch_time, 2),
        "merge_time_s": round(merge_time, 2),
        "upload_time_s": round(ul_time, 2),
        "chunk_size": chunk_size,
        "final_mosaic_path": str(final_mosaic),
        "final_mosaic_exists": final_mosaic.exists(),
        "final_mosaic_size_bytes": final_mosaic.stat().st_size if final_mosaic.exists() else 0,
    }

    # Try to read canvas dimensions from the final mosaic
    try:
        import cv2
        img = cv2.imread(str(final_mosaic))
        if img is not None:
            metrics["canvas_width"] = img.shape[1]
            metrics["canvas_height"] = img.shape[0]
            metrics["canvas_mpx"] = round(img.shape[0] * img.shape[1] / 1e6, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            metrics["content_pct"] = round(100.0 * (gray > 10).mean(), 1)
    except Exception as e:
        metrics["image_read_error"] = str(e)

    # Cleanup
    shutil.rmtree(work, ignore_errors=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-input", required=True, help="S3 prefix with input images")
    parser.add_argument("--s3-output", required=True, help="S3 prefix for output")
    parser.add_argument("--chunk-size", type=int, default=18, help="frames per chunk")
    parser.add_argument("--work-dir", default="/tmp/scratchpad", help="local scratch directory")
    args = parser.parse_args()

    metrics = run_pipeline(args.s3_input, args.s3_output, args.chunk_size, args.work_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()