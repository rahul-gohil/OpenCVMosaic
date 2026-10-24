#!/usr/bin/env python3
"""
Pipeline MCP Server - Runs on EC2 instances.
Receives pipeline job specs via HTTP POST and runs the orthomosaic pipeline.

Supports:
- Async pipeline execution with status polling
- Real-time image serving (generated strips + merged result)
- Per-stage timing extraction
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
import base64
import threading
import glob
import csv
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/pipeline-mcp.log'),
    ],
)
logger = logging.getLogger("pipeline-mcp-server")

try:
    import cv2
    OPENCV_AVAILABLE = True
    logger.info(f"OpenCV {cv2.__version__} loaded from {cv2.__file__}")
except ImportError as e:
    OPENCV_AVAILABLE = False
    logger.error(f"OpenCV not available: {e}")

PIPELINE_DIR = os.environ.get("PIPELINE_DIR", "/opt/OpenCVMosaic")
PYTHON = os.environ.get("PYTHON", "python3")

# Global state for async pipeline execution
pipeline_state = {
    "status": "idle",  # idle, running, completed, failed
    "progress": {"current_step": "Idle", "progress_percent": 0},
    "metrics": None,
    "error": None,
    "work_dir": None,
    "start_time": None,
    "thread": None,
}


def _run_pipeline_sync(chunk_size, work_dir):
    """Run the pipeline synchronously (called in a background thread)."""
    global pipeline_state
    pipeline_state["status"] = "running"
    pipeline_state["start_time"] = time.perf_counter()
    pipeline_state["work_dir"] = work_dir

    try:
        # Step 1: Use local assets (shipped with repo)
        input_dir = os.path.join(PIPELINE_DIR, "COOL-Benchmark", "assets")
        if not os.path.isdir(input_dir):
            raise Exception(f"Assets directory not found: {input_dir}")
        pipeline_state["progress"] = {"current_step": "Loading local assets", "progress_percent": 5}
        dl_time = 0.0

        # Step 2: Run chunked stitching
        pipeline_state["progress"] = {"current_step": "Stitching chunks", "progress_percent": 15}
        stitch_start = time.perf_counter()
        proc = subprocess.run(
            [PYTHON, "main.py", "run-chunks",
             "--dataset", input_dir,
             "--pattern", "*.jpg",
             "--chunk-size", str(chunk_size),
             "--output-root", os.path.join(work_dir, "chunks"),
             "--prefix", "chunk"],
            cwd=PIPELINE_DIR, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            logger.error(f"Stitch failed: {proc.stderr[-500:]}")
            raise Exception(f"Stitch failed: {proc.stderr[-500:]}")
        stitch_time = time.perf_counter() - stitch_start
        pipeline_state["progress"] = {"current_step": "Stitching complete", "progress_percent": 50}

        # Step 3: Run merge
        pipeline_state["progress"] = {"current_step": "Merging chunks", "progress_percent": 55}
        merge_start = time.perf_counter()
        chunk_pattern = os.path.join(work_dir, "chunks", "chunk_*", "mosaic_final.jpg")
        proc = subprocess.run(
            [PYTHON, "main.py", "merge",
             "--inputs", chunk_pattern,
             "--out", os.path.join(work_dir, "merged")],
            cwd=PIPELINE_DIR, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            logger.error(f"Merge failed: {proc.stderr[-500:]}")
            raise Exception(f"Merge failed: {proc.stderr[-500:]}")
        merge_time = time.perf_counter() - merge_start
        # Read per-stage merge timings
        merge_timings_path = os.path.join(work_dir, "merged", "merge_timings.json")
        merge_timings = {}
        if os.path.exists(merge_timings_path):
            try:
                with open(merge_timings_path) as f:
                    merge_timings = json.load(f)
            except Exception:
                pass
        # Generate downscaled preview of merged result
        merged_path = os.path.join(work_dir, "merged", "mosaic_merged.jpg")
        preview_path = os.path.join(work_dir, "merged", "mosaic_merged_preview.jpg")
        if os.path.exists(merged_path):
            try:
                img = cv2.imread(merged_path)
                if img is not None:
                    h, w = img.shape[:2]
                    preview = cv2.resize(img, (w // 10, h // 10), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    logger.info(f"Generated preview: {preview_path}")
            except Exception as e:
                logger.warning(f"Could not generate preview: {e}")
        pipeline_state["progress"] = {"current_step": "Merge complete", "progress_percent": 85}

        # Step 4: Skip upload (no S3) - results available via image gallery
        pipeline_state["progress"] = {"current_step": "Finalizing", "progress_percent": 90}
        ul_time = 0.0

        total_time = time.perf_counter() - pipeline_state["start_time"]

        # Collect per-stage timings from timings.csv files
        per_stage = _collect_per_stage_timings(work_dir)

        # Collect metrics
        merged_dir = os.path.join(work_dir, "merged")
        final_mosaic = os.path.join(merged_dir, "mosaic_merged.jpg")

        metrics = {
            "total_time_s": round(total_time, 2),
            "stitch_time_s": round(stitch_time, 2),
            "merge_time_s": round(merge_time, 2),
            "merge_stage_timings": merge_timings,
            "chunk_size": chunk_size,
            "opencv_version": cv2.__version__ if OPENCV_AVAILABLE else "unknown",
            "is_cool": "/opt/cool" in (cv2.__file__ if OPENCV_AVAILABLE else ""),
            "final_mosaic_exists": os.path.exists(final_mosaic),
            "final_mosaic_size_bytes": os.path.getsize(final_mosaic) if os.path.exists(final_mosaic) else 0,
            "per_stage_timings": per_stage,
        }

        if os.path.exists(final_mosaic):
            img = cv2.imread(final_mosaic)
            if img is not None:
                metrics["canvas_width"] = img.shape[1]
                metrics["canvas_height"] = img.shape[0]
                metrics["canvas_mpx"] = round(img.shape[0] * img.shape[1] / 1e6, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                metrics["content_pct"] = round(100.0 * (gray > 10).mean(), 1)

        pipeline_state["metrics"] = metrics
        pipeline_state["progress"] = {"current_step": "Completed", "progress_percent": 100}
        pipeline_state["status"] = "completed"
        logger.info(f"Pipeline complete: {metrics}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        pipeline_state["status"] = "failed"
        pipeline_state["error"] = str(e)
        pipeline_state["progress"] = {"current_step": "Failed", "progress_percent": 0}

    finally:
        # Cleanup work dir after a delay (keep images for viewing)
        pass


def _collect_per_stage_timings(work_dir):
    """Read timings.csv from each chunk and aggregate per-stage times."""
    timings = {"sift": 0.0, "matching": 0.0, "homography": 0.0, "stitching": 0.0, "frames": 0}
    chunks_dir = os.path.join(work_dir, "chunks")
    for csv_path in glob.glob(os.path.join(chunks_dir, "chunk_*", "timings.csv")):
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timings["sift"] += float(row.get("sift_time", 0))
                    timings["matching"] += float(row.get("match_time", 0))
                    timings["homography"] += float(row.get("homography_time", 0))
                    timings["stitching"] += float(row.get("stitch_time", 0))
                    timings["frames"] += 1
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
    # Round values
    for k in ["sift", "matching", "homography", "stitching"]:
        timings[k] = round(timings[k], 2)
    return timings


def _list_generated_images(work_dir):
    """List generated chunk images and merged result."""
    images = {"chunks": [], "merged": None}
    if not work_dir:
        return images
    chunks_dir = os.path.join(work_dir, "chunks")
    for chunk_dir in sorted(glob.glob(os.path.join(chunks_dir, "chunk_*"))):
        mosaic_path = os.path.join(chunk_dir, "mosaic_final.jpg")
        if os.path.exists(mosaic_path):
            images["chunks"].append({
                "name": os.path.basename(chunk_dir),
                "path": mosaic_path,
                "size_bytes": os.path.getsize(mosaic_path),
            })
    merged_path = os.path.join(work_dir, "merged", "mosaic_merged.jpg")
    preview_path = os.path.join(work_dir, "merged", "mosaic_merged_preview.jpg")
    if os.path.exists(merged_path):
        images["merged"] = {
            "path": merged_path,
            "size_bytes": os.path.getsize(merged_path),
        }
    if os.path.exists(preview_path):
        images["merged_preview"] = {
            "path": preview_path,
            "size_bytes": os.path.getsize(preview_path),
        }
    return images


async def handle_health(request):
    return web.json_response({
        "status": "healthy",
        "opencv_available": OPENCV_AVAILABLE,
        "opencv_version": cv2.__version__ if OPENCV_AVAILABLE else None,
        "opencv_path": cv2.__file__ if OPENCV_AVAILABLE else None,
        "is_cool": "/opt/cool" in (cv2.__file__ if OPENCV_AVAILABLE else ""),
        "pipeline_status": pipeline_state["status"],
    })


async def handle_start_pipeline(request):
    global pipeline_state
    if pipeline_state["status"] == "running":
        return web.json_response({"error": "Pipeline already running"}, status=409)

    data = await asyncio.wait_for(request.json(), timeout=60)
    chunk_size = data.get("chunk_size", 18)
    work_dir = data.get("work_dir", "/tmp/scratchpad")

    # Reset state
    pipeline_state = {
        "status": "starting",
        "progress": {"current_step": "Initializing", "progress_percent": 0},
        "metrics": None,
        "error": None,
        "work_dir": work_dir,
        "start_time": None,
        "thread": None,
    }

    # Clean work dir
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    # Start pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(chunk_size, work_dir),
        daemon=True,
    )
    pipeline_state["thread"] = thread
    thread.start()

    return web.json_response({"status": "started", "work_dir": work_dir})


async def handle_pipeline_status(request):
    images = _list_generated_images(pipeline_state.get("work_dir"))
    return web.json_response({
        "status": pipeline_state["status"],
        "progress": pipeline_state["progress"],
        "metrics": pipeline_state["metrics"],
        "error": pipeline_state["error"],
        "images": images,
        "elapsed_s": round(time.perf_counter() - pipeline_state["start_time"], 1) if pipeline_state["start_time"] else 0,
    })


async def handle_pipeline_image(request):
    """Serve an image file from the instance."""
    img_path = request.query.get("path", "")
    if not img_path or not os.path.exists(img_path):
        return web.json_response({"error": "Image not found"}, status=404)
    # Security: only serve from work_dir
    work_dir = pipeline_state.get("work_dir", "")
    if work_dir and not img_path.startswith(work_dir):
        return web.json_response({"error": "Access denied"}, status=403)
    return web.FileResponse(img_path)


async def main():
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_post("/run-pipeline", handle_start_pipeline)
    app.router.add_get("/pipeline-status", handle_pipeline_status)
    app.router.add_get("/pipeline-image", handle_pipeline_image)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logger.info("Pipeline MCP server running on port 8080")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
