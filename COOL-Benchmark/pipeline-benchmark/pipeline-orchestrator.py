#!/usr/bin/env python3
"""
Pipeline Benchmark Orchestrator

Simplified orchestrator for the orthomosaic pipeline benchmark.
Reuses the instance manager and build manager from COOL-Benchmark,
but replaces the findContours MCP server with the pipeline runner.

Endpoints (same port 8080 as the original orchestrator):
    GET  /health                    - Health check
    POST /api/benchmark/run         - Start a pipeline benchmark
    GET  /api/benchmark/{id}/status  - Get benchmark status
    POST /api/instances/cleanup     - Cleanup orphaned instances

Usage:
    python pipeline-orchestrator.py
    # Then open frontend at http://localhost:3000
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import aiohttp
from aiohttp import web

# Add COOL-Benchmark paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentcore"))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "instance_manager",
    os.path.join(os.path.dirname(__file__), "..", "agentcore", "instance-manager.py"),
)
instance_manager_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instance_manager_module)
InstanceManager = instance_manager_module.InstanceManager

from build_manager import BuildManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline-orchestrator.log")],
)
logger = logging.getLogger("pipeline-orchestrator")
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


class TaskStatus(Enum):
    PENDING = "pending"
    STAGING = "staging"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineBenchmarkTask:
    task_id: str
    instance_type: str
    max_instances: int
    chunk_size: int
    status: TaskStatus
    start_time: float
    opencv_version: str = "5"
    mode: str = "diy"
    current_run: str = "cool"  # "cool", "diy", or "completed"
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    ssh_user: str = "ec2-user"
    local_images: Optional[Dict[str, str]] = None
    cool_local_images: Optional[Dict[str, str]] = None
    build_progress: Optional[Dict[str, Any]] = None
    end_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineOrchestrator:
    def __init__(self, region: str = None, marketplace_ami_id: str = None, diy_ami_id: str = None):
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-1"))
        self.marketplace_ami_id = marketplace_ami_id or os.environ.get("MARKETPLACE_AMI_ID", "")
        # Standard Graviton AMI for DIY-only mode (no COOL subscription needed)
        self.diy_ami_id = diy_ami_id or os.environ.get("DIY_AMI_ID", "ami-0373a05f3fe4e2663")
        self.instance_manager = InstanceManager(region=self.region)
        self.build_manager = BuildManager(region=self.region)
        self.tasks: Dict[str, PipelineBenchmarkTask] = {}
        self.default_region = self.region
        self._last_pipeline_images = {}
        self._cool_pipeline_images = {}
        self._last_pipeline_stage = ""
        self._last_pipeline_pct = 0

        # Load from config-marketplace.json if available
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config-marketplace.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                    marketplace = config.get("marketplace", {})
                    if not self.marketplace_ami_id:
                        self.marketplace_ami_id = marketplace.get("ami_id", self.marketplace_ami_id)
                    diy = config.get("diy_ami", {})
                    if diy.get("ami_id") and not os.environ.get("DIY_AMI_ID"):
                        self.diy_ami_id = diy["ami_id"]
                    logger.info(f"Loaded marketplace config: COOL AMI={self.marketplace_ami_id}")
        except Exception as e:
            logger.warning(f"Could not load config-marketplace.json: {e}")
        logger.info(f"DIY AMI (standard Graviton): {self.diy_ami_id}")

    async def initialize(self):
        try:
            await self.instance_manager.initialize()
        except Exception as e:
            logger.warning(f"Instance manager init failed (no AWS creds?): {e}")
            logger.warning("Orchestrator will run in offline mode - AWS operations will fail")
        logger.info("Pipeline orchestrator initialized")

    async def start_benchmark(
        self, instance_type: str, max_instances: int, chunk_size: int,
        opencv_version: str = "5",
        mode: str = "diy",
    ) -> str:
        task_id = str(__import__("uuid").uuid4())
        task = PipelineBenchmarkTask(
            task_id=task_id,
            instance_type=instance_type,
            max_instances=max_instances,
            chunk_size=chunk_size,
            status=TaskStatus.PENDING,
            start_time=__import__("time").time(),
        )
        task.opencv_version = opencv_version
        task.mode = mode
        task.build_progress = {"current_step": "Initializing", "progress_percent": 0}
        self.tasks[task_id] = task
        asyncio.create_task(self._execute_benchmark(task_id))
        logger.info(f"Started pipeline benchmark {task_id}")
        return task_id

    async def _execute_benchmark(self, task_id: str):
        task = self.tasks[task_id]
        import time
        start_time = time.time()
        instance_id = None

        try:
            task.status = TaskStatus.STAGING
            task.build_progress = {"current_step": "Launching instance", "progress_percent": 10}

            # Determine AMI and user data based on mode
            mode = getattr(task, "mode", "diy")
            if mode == "diy":
                # DIY-only: use standard Graviton AMI with pip OpenCV
                ami_id = self.diy_ami_id
                user_data = self._get_diy_user_data()
                task.ssh_user = "ec2-user"
                logger.info(f"DIY mode: using standard Graviton AMI {ami_id}")
            elif mode == "cool":
                # COOL-only: use COOL marketplace AMI
                ami_id = self.marketplace_ami_id
                if not ami_id or ami_id.startswith("ami-to-be"):
                    raise ValueError("Marketplace AMI not configured. Set MARKETPLACE_AMI_ID or config-marketplace.json")
                user_data = self._get_user_data()
                task.ssh_user = "ubuntu"
                logger.info(f"COOL mode: using marketplace AMI {ami_id}")
            else:
                # Compare mode: use COOL marketplace AMI (runs both)
                ami_id = self.marketplace_ami_id
                if not ami_id or ami_id.startswith("ami-to-be"):
                    raise ValueError("Marketplace AMI not configured. Set MARKETPLACE_AMI_ID or config-marketplace.json")
                user_data = self._get_user_data()
                task.ssh_user = "ubuntu"
                logger.info(f"Compare mode: using marketplace AMI {ami_id}")

            # Launch instance
            instance_id = await self.instance_manager.launch_instance(
                instance_type=task.instance_type,
                ami_id=ami_id,
                user_data=user_data,
                tags={
                    "Name": f"pipeline-benchmark-{task_id[:8]}",
                    "BenchmarkTaskId": task_id,
                    "AutoTerminate": "true",
                    "Mode": mode,
                },
            )
            task.instance_id = instance_id
            await self.instance_manager.wait_for_instance_ready(instance_id)
            instance = self.instance_manager.instances.get(instance_id)
            ip = instance.public_ip if instance.public_ip else instance.private_ip
            task.instance_ip = ip

            task.build_progress = {"current_step": "Waiting for pipeline server", "progress_percent": 30}

            # Wait for pipeline MCP server to be ready
            mcp_ready = False
            for i in range(60):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://{ip}:8080/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                mcp_ready = True
                                break
                except Exception:
                    await asyncio.sleep(10)

            if not mcp_ready:
                raise Exception("Pipeline MCP server not ready after 10 minutes")

            task.status = TaskStatus.RUNNING
            task.build_progress = {"current_step": "Running pipeline", "progress_percent": 50}

            mode = getattr(task, "mode", "diy")

            if mode == "diy":
                task.current_run = "diy"
                # DIY-only: run once with standard OpenCV
                diy_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size, work_dir="/tmp/scratchpad_diy"
                )
                task.build_progress = {"current_step": "Downloading results", "progress_percent": 90}
                task.status = TaskStatus.DOWNLOADING
                # Download results locally before marking complete
                ip = getattr(task, "instance_ip", None)
                if ip:
                    await self._download_results_locally(task, ip)
                task.results = {
                    "diy": diy_metrics,
                    "mode": "diy",
                }
                task.status = TaskStatus.COMPLETED

            elif mode == "cool":
                task.current_run = "cool"
                # COOL-only: run once with COOL OpenCV
                cool_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size, work_dir="/tmp/scratchpad_cool"
                )
                task.build_progress = {"current_step": "Downloading results", "progress_percent": 90}
                task.status = TaskStatus.DOWNLOADING
                ip = getattr(task, "instance_ip", None)
                if ip:
                    await self._download_results_locally(task, ip)
                task.results = {
                    "cool": cool_metrics,
                    "mode": "cool",
                }
                task.status = TaskStatus.COMPLETED

            else:
                # Compare mode: run both COOL and DIY
                task.current_run = "cool"
                # Run COOL (marketplace) build first
                cool_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size, work_dir="/tmp/scratchpad_cool"
                )

                # Save COOL images before they're overwritten
                self._cool_pipeline_images = getattr(self, "_last_pipeline_images", {}).copy()
                # Reconfigure to DIY (pip)
                task.current_run = "reconfiguring"
                task.build_progress = {"current_step": "Reconfiguring to DIY", "progress_percent": 70}
                await self._reconfigure_to_pip(ip)

                # Run DIY build
                task.current_run = "diy"
                diy_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size, work_dir="/tmp/scratchpad_diy"
                )

                # Compare
                task.build_progress = {"current_step": "Downloading results", "progress_percent": 90}
                task.status = TaskStatus.DOWNLOADING
                ip = getattr(task, "instance_ip", None)
                if ip:
                    await self._download_results_locally(task, ip)
                task.current_run = "completed"
                task.results = {
                    "cool": cool_metrics,
                    "diy": diy_metrics,
                    "comparison": self._compare(cool_metrics, diy_metrics),
                    "mode": "compare",
                }
                task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            task.build_progress = {"current_step": "Completed", "progress_percent": 100}

        except Exception as e:
            logger.error(f"Benchmark {task_id} failed: {e}", exc_info=True)
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = time.time()
        finally:
            if instance_id:
                logger.info(f"Instance {instance_id} left running for debugging (auto-terminate disabled)")

    async def _run_pipeline_on_instance(self, ip: str, chunk_size: int, work_dir: str = None) -> dict:
        import asyncio as _asyncio
        async with aiohttp.ClientSession() as session:
            # Start pipeline (returns immediately now)
            if work_dir is None:
                work_dir = "/tmp/scratchpad"
            async with session.post(
                f"http://{ip}:8080/run-pipeline",
                json={"chunk_size": chunk_size, "work_dir": work_dir},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    raise Exception(f"Failed to start pipeline: {text}")

            # Poll for completion with exponential backoff
            consecutive_errors = 0
            poll_interval = 5
            while True:
                await _asyncio.sleep(poll_interval)
                try:
                    async with session.get(
                        f"http://{ip}:8080/pipeline-status",
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        data = await resp.json()
                        consecutive_errors = 0
                        poll_interval = 5  # reset on success
                        status = data.get("status", "unknown")
                        progress = data.get("progress", {})
                        self._last_pipeline_images = data.get("images", {})
                        self._last_pipeline_stage = progress.get("current_step", "")
                        self._last_pipeline_pct = progress.get("progress_percent", 0)

                        if status == "completed":
                            metrics = data.get("metrics", {})
                            metrics["per_stage_timings"] = metrics.get("per_stage_timings", {})
                            metrics["merge_stage_timings"] = metrics.get("merge_stage_timings", {})
                            return metrics
                        elif status == "failed":
                            raise Exception(data.get("error", "Pipeline failed"))
                except Exception as e:
                    consecutive_errors += 1
                    poll_interval = min(poll_interval * 2, 20)  # exponential backoff: 5→10→20
                    logger.warning(f"Poll error ({consecutive_errors}/5): {e}, next poll in {poll_interval}s")
                    if consecutive_errors >= 5:
                        raise Exception(f"Lost connection to instance after 5 retries: {e}")

    async def _reconfigure_to_pip(self, ip: str):
        """Reconfigure from COOL to DIY using system Python + pip OpenCV.

        Uses SSH (not SSM). Avoids heredoc-in-bash-c issues by writing
        the systemd service file via python3 -c instead of cat heredoc.
        """
        pip_version = "5.0.0.93"

        # Build the systemd service file content as a base64-encoded string
        # to avoid all quoting/heredoc issues over SSH
        import base64 as _b64
        service_content = """[Unit]
Description=Pipeline MCP Server (DIY pip)
After=network.target

[Service]
Type=simple
User=ubuntu
Environment=PIPELINE_DIR=/opt/OpenCVMosaic
ExecStart=/usr/bin/python3 /opt/pipeline-mcp-server.py
Restart=always
RestartSec=5
StandardOutput=append:/tmp/pipeline-mcp.log
StandardError=append:/tmp/pipeline-mcp.log

[Install]
WantedBy=multi-user.target
"""
        service_b64 = _b64.b64encode(service_content.encode()).decode()

        commands = [
            # Step 1: Stop COOL MCP server
            'echo "Step 1: Stopping COOL MCP server..."',
            "sudo systemctl stop pipeline-mcp 2>/dev/null || true",
            "sleep 2",

            # Step 2: Install pip + OpenCV via system Python
            'echo "Step 2: Installing pip + OpenCV via system Python..."',
            "sudo apt-get update -y -qq",
            "sudo apt-get install -y -qq python3-pip",
            f"sudo pip3 install --break-system-packages opencv-python-headless=={pip_version} numpy scipy aiohttp || exit 1",

            # Step 3: Register bundled wheel libs + fix cv2 bootstrap
            (
                'echo "Step 3: Registering bundled wheel libs and fixing cv2 bootstrap..." && '
                'SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])") && '
                'LDCONF=/etc/ld.so.conf.d/pip-wheel-bundled.conf && '
                '> "$LDCONF" && '
                'for D in "$SITE_PKG/numpy.libs" "$SITE_PKG/opencv_python_headless.libs"; do '
                '  [ -d "$D" ] && echo "$D" >> "$LDCONF" && echo "Registered: $D"; '
                'done; '
                'sudo ldconfig && '
                'CV2_DIR="$SITE_PKG/cv2" && '
                'ABI3="$CV2_DIR/cv2.abi3.so" && '
                'PY312="$CV2_DIR/python-3.12" && '
                '[ -f "$PY312/cv2.cpython-312-aarch64-linux-gnu.so" ] && '
                '  rm -f "$PY312/cv2.cpython-312-aarch64-linux-gnu.so" && '
                '  echo "Removed stale cpython-312 extension"; '
                'find "$CV2_DIR" -maxdepth 1 -name "cv2.cpython-*.so" -delete 2>/dev/null; '
                '[ -f "$ABI3" ] && mkdir -p "$PY312" && ln -sf "$ABI3" "$PY312/cv2.abi3.so" && '
                '  echo "Linked abi3 into python-3.12/"; '
                'python3 -c "import cv2; print(f\"DIY OpenCV {cv2.__version__} installed\")" || exit 1'
            ),

            # Step 4: Write systemd service file via base64 (no heredoc issues)
            'echo "Step 4: Switching systemd to DIY (system Python)..."',
            f'echo "{service_b64}" | base64 -d | sudo tee /etc/systemd/system/pipeline-mcp.service > /dev/null',
            "sudo systemctl daemon-reload",

            # Step 5: Restart with DIY config
            'echo "Step 5: Starting DIY MCP server..."',
            "sudo systemctl restart pipeline-mcp 2>/dev/null || true",
            "sleep 5",

            # Step 6: Verify
            'echo "Step 6: Verifying DIY MCP server..."',
            'for i in $(seq 1 30); do STATUS=$(systemctl is-active pipeline-mcp 2>/dev/null || true); if [ "$STATUS" = "active" ]; then echo "MCP service active"; break; fi; echo "Waiting for MCP service... ($STATUS, ${i}s)"; sleep 1; done',
            'curl -s http://localhost:8080/health || echo "MCP not yet responding"',
            'echo "DIY reconfiguration complete"',
        ]

        import subprocess
        key_path = os.environ.get("EC2_KEY_PATH", "key.pem")
        ssh_user = "ubuntu"
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                   f"{ssh_user}@{ip}", " && ".join(commands)]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Reconfigure stdout: {result.stdout}")
            logger.error(f"Reconfigure stderr: {result.stderr}")
            raise Exception(f"DIY reconfiguration failed: {result.stderr[:500]}")

        # Wait for MCP server to be ready in DIY mode
        import asyncio
        for i in range(30):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://{ip}:8080/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if not data.get("is_cool", True):
                                logger.info(f"MCP server ready in DIY mode (OpenCV {data.get('opencv_version')}) on {ip}")
                                return
                            else:
                                logger.warning(f"MCP server still in COOL mode, retrying... (is_cool={data.get('is_cool')})")
            except Exception:
                pass
            await asyncio.sleep(5)
        raise Exception("MCP server not ready in DIY mode after reconfigure")

    async def _download_results_locally(self, task, ip: str):
        """Download pipeline results from instance to local machine before termination."""
        import subprocess, os as _os, tempfile
        key_path = os.environ.get("EC2_KEY_PATH", "key.pem")
        ssh_user = getattr(task, "ssh_user", "ec2-user")
        local_dir = os.path.join(tempfile.gettempdir(), f"pipeline-results-{task.task_id}")
        os.makedirs(local_dir, exist_ok=True)
        task.local_images = {"chunks": [], "merged": None, "merged_preview": None}

        images = getattr(self, "_last_pipeline_images", {})
        if not images:
            return

        # Download chunk images
        for chunk in images.get("chunks", []):
            remote_path = chunk["path"]
            local_name = f"{chunk['name']}.jpg"
            local_path = os.path.join(local_dir, local_name)
            try:
                proc = subprocess.run(
                    ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                     f"{ssh_user}@{ip}:{remote_path}", local_path],
                    capture_output=True, text=True, timeout=60
                )
                if proc.returncode == 0 and _os.path.exists(local_path):
                    task.local_images["chunks"].append({
                        "name": chunk["name"],
                        "path": local_path,
                        "size_bytes": _os.path.getsize(local_path),
                    })
            except Exception as e:
                logger.warning(f"Could not download {remote_path}: {e}")

        # Download merged result
        if images.get("merged"):
            remote_path = images["merged"]["path"]
            local_path = os.path.join(local_dir, "mosaic_merged.jpg")
            try:
                proc = subprocess.run(
                    ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                     f"{ssh_user}@{ip}:{remote_path}", local_path],
                    capture_output=True, text=True, timeout=120
                )
                if proc.returncode == 0 and _os.path.exists(local_path):
                    task.local_images["merged"] = {
                        "path": local_path,
                        "size_bytes": _os.path.getsize(local_path),
                    }
            except Exception as e:
                logger.warning(f"Could not download merged: {e}")

        # Download preview
        if images.get("merged_preview"):
            remote_path = images["merged_preview"]["path"]
            local_path = os.path.join(local_dir, "mosaic_merged_preview.jpg")
            try:
                proc = subprocess.run(
                    ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                     f"{ssh_user}@{ip}:{remote_path}", local_path],
                    capture_output=True, text=True, timeout=60
                )
                if proc.returncode == 0 and _os.path.exists(local_path):
                    task.local_images["merged_preview"] = {
                        "path": local_path,
                        "size_bytes": _os.path.getsize(local_path),
                    }
            except Exception as e:
                logger.warning(f"Could not download preview: {e}")

        logger.info(f"Downloaded {len(task.local_images['chunks'])} chunks + merged to {local_dir}")

        # Download COOL images (if available from compare mode)
        cool_images = getattr(self, "_cool_pipeline_images", {})
        if cool_images:
            cool_local_dir = os.path.join(tempfile.gettempdir(), f"pipeline-results-{task.task_id}-cool")
            os.makedirs(cool_local_dir, exist_ok=True)
            task.cool_local_images = {"chunks": [], "merged": None, "merged_preview": None}

            for chunk in cool_images.get("chunks", []):
                remote_path = chunk["path"]
                local_name = f"{chunk['name']}.jpg"
                local_path = os.path.join(cool_local_dir, local_name)
                try:
                    proc = subprocess.run(
                        ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                         f"{ssh_user}@{ip}:{remote_path}", local_path],
                        capture_output=True, text=True, timeout=60
                    )
                    if proc.returncode == 0 and _os.path.exists(local_path):
                        task.cool_local_images["chunks"].append({
                            "name": chunk["name"],
                            "path": local_path,
                            "size_bytes": _os.path.getsize(local_path),
                        })
                except Exception as e:
                    logger.warning(f"Could not download COOL {remote_path}: {e}")

            if cool_images.get("merged"):
                remote_path = cool_images["merged"]["path"]
                local_path = os.path.join(cool_local_dir, "mosaic_merged.jpg")
                try:
                    proc = subprocess.run(
                        ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                         f"{ssh_user}@{ip}:{remote_path}", local_path],
                        capture_output=True, text=True, timeout=120
                    )
                    if proc.returncode == 0 and _os.path.exists(local_path):
                        task.cool_local_images["merged"] = {
                            "path": local_path,
                            "size_bytes": _os.path.getsize(local_path),
                        }
                except Exception as e:
                    logger.warning(f"Could not download COOL merged: {e}")

            if cool_images.get("merged_preview"):
                remote_path = cool_images["merged_preview"]["path"]
                local_path = os.path.join(cool_local_dir, "mosaic_merged_preview.jpg")
                try:
                    proc = subprocess.run(
                        ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                         f"{ssh_user}@{ip}:{remote_path}", local_path],
                        capture_output=True, text=True, timeout=60
                    )
                    if proc.returncode == 0 and _os.path.exists(local_path):
                        task.cool_local_images["merged_preview"] = {
                            "path": local_path,
                            "size_bytes": _os.path.getsize(local_path),
                        }
                except Exception as e:
                    logger.warning(f"Could not download COOL preview: {e}")

            logger.info(f"Downloaded {len(task.cool_local_images['chunks'])} COOL chunks + merged")

    def _compare(self, cool: dict, diy: dict) -> dict:
        cool_time = cool.get("total_time_s", 0)
        diy_time = diy.get("total_time_s", 0)
        speedup = diy_time / cool_time if cool_time > 0 and diy_time > 0 else None
        return {
            "speedup": round(speedup, 2) if speedup else None,
            "time_saved_s": round(diy_time - cool_time, 1) if cool_time > 0 and diy_time > 0 else None,
            "cool_total_time_s": cool_time,
            "diy_total_time_s": diy_time,
            "cool_stitch_time_s": cool.get("stitch_time_s"),
            "diy_stitch_time_s": diy.get("stitch_time_s"),
            "cool_merge_time_s": cool.get("merge_time_s"),
            "diy_merge_time_s": diy.get("merge_time_s"),
            "cool_canvas_mpx": cool.get("canvas_mpx"),
            "diy_canvas_mpx": diy.get("canvas_mpx"),
            "cool_content_pct": cool.get("content_pct"),
            "diy_content_pct": diy.get("content_pct"),
        }

    def _get_user_data(self) -> str:
        return """#!/bin/bash
# COOL AMI (Ubuntu 24.04, Graviton5) - has Python and git pre-installed
cd /opt
if [ ! -d "/opt/OpenCVMosaic" ]; then
    git clone --depth 1 https://github.com/rahul-gohil/OpenCVMosaic.git /opt/OpenCVMosaic
fi
cd /opt/OpenCVMosaic
# Install aiohttp in COOL venv (for MCP server)
/opt/cool/venvs/python_3.12/bin/pip install --quiet numpy scipy aiohttp 2>/dev/null || true
# Ensure python3-pip is available for DIY comparison mode (installed via reconfigure if needed)
apt-get update -y -qq 2>/dev/null || true
apt-get install -y -qq python3-pip 2>/dev/null || true
cp /opt/OpenCVMosaic/COOL-Benchmark/opencv-ami/pipeline-mcp-server.py /opt/pipeline-mcp-server.py
# Create wrapper script that uses COOL venv Python with COOL's OpenCV
cat > /opt/run-mcp.sh << 'WRAPPER'
#!/bin/bash
export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/cool/python_3.12/site-packages/:$PYTHONPATH
exec /opt/cool/venvs/python_3.12/bin/python /opt/pipeline-mcp-server.py
WRAPPER
chmod +x /opt/run-mcp.sh
# Start the pipeline MCP server as a systemd service
cat > /etc/systemd/system/pipeline-mcp.service << 'EOF'
[Unit]
Description=Pipeline MCP Server
After=network.target

[Service]
Type=simple
User=ubuntu
Environment=PIPELINE_DIR=/opt/OpenCVMosaic
Environment=LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/ffmpeg/lib
Environment=PYTHONPATH=/opt/cool/python_3.12/site-packages/
ExecStart=/opt/run-mcp.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable pipeline-mcp
systemctl start pipeline-mcp
echo "Pipeline ready" > /tmp/pipeline_ready
"""

    def _get_diy_user_data(self) -> str:
        """User data for standard Graviton AMI (no COOL) - installs pip OpenCV."""
        return """#!/bin/bash
set -e
# Update system
dnf update -y || apt-get update -y

# Install Python, pip, git
if command -v dnf &> /dev/null; then
    dnf install -y python3 python3-pip git
else
    apt-get install -y python3 python3-pip git
fi

# Install OpenCV and dependencies via pip
pip3 install --break-system-packages opencv-python-headless numpy scipy aiohttp psutil Pillow 2>/dev/null || pip3 install opencv-python-headless numpy scipy aiohttp psutil Pillow

# Clone the pipeline code from GitHub (no AWS credentials needed)
mkdir -p /opt
cd /opt
if [ ! -d "/opt/OpenCVMosaic" ]; then
    git clone --depth 1 https://github.com/rahul-gohil/OpenCVMosaic.git /opt/OpenCVMosaic
fi
cd /opt/OpenCVMosaic

# Deploy pipeline MCP server
cp /opt/OpenCVMosaic/COOL-Benchmark/opencv-ami/pipeline-mcp-server.py /opt/pipeline-mcp-server.py
chmod +x /opt/pipeline-mcp-server.py

# Start the pipeline MCP server as a systemd service
PYTHON=$(which python3)
cat > /etc/systemd/system/pipeline-mcp.service << EOF
[Unit]
Description=Pipeline MCP Server (DIY OpenCV)
After=network.target

[Service]
Type=simple
User=root
Environment=PYTHON=$PYTHON
Environment=PIPELINE_DIR=/opt/OpenCVMosaic
ExecStart=$PYTHON /opt/pipeline-mcp-server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable pipeline-mcp
systemctl start pipeline-mcp
echo "DIY Pipeline ready" > /tmp/pipeline_ready
"""

    async def get_benchmark_status(self, task_id: str) -> dict:
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}
        ip = getattr(task, "instance_ip", None)
        pipeline_images = getattr(self, "_last_pipeline_images", {})
        cool_pipeline_images = getattr(self, "_cool_pipeline_images", {})
        local_images = getattr(task, "local_images", None)
        cool_local_images = getattr(task, "cool_local_images", None)
        # Build image URLs - prefer local (downloaded) over remote (SCP)
        image_urls = {"chunks": [], "merged": None, "merged_preview": None}
        if local_images:
            for chunk in local_images.get("chunks", []):
                image_urls["chunks"].append({
                    "name": chunk["name"],
                    "url": f"/api/benchmark/{task.task_id}/local-image?name={chunk['name']}.jpg",
                    "size_bytes": chunk.get("size_bytes", 0),
                })
            if local_images.get("merged"):
                image_urls["merged"] = {
                    "url": f"/api/benchmark/{task.task_id}/local-image?name=mosaic_merged.jpg",
                    "size_bytes": local_images["merged"].get("size_bytes", 0),
                    "download_url": f"/api/benchmark/{task.task_id}/local-image?name=mosaic_merged.jpg&download=1",
                }
            if local_images.get("merged_preview"):
                image_urls["merged_preview"] = {
                    "url": f"/api/benchmark/{task.task_id}/local-image?name=mosaic_merged_preview.jpg",
                    "size_bytes": local_images["merged_preview"].get("size_bytes", 0),
                }
        # Build COOL local image URLs (if downloaded)
        cool_local_image_urls = {"chunks": [], "merged": None, "merged_preview": None}
        if cool_local_images:
            for chunk in cool_local_images.get("chunks", []):
                cool_local_image_urls["chunks"].append({
                    "name": chunk["name"],
                    "url": f"/api/benchmark/{task.task_id}/cool-image?name={chunk['name']}.jpg",
                    "size_bytes": chunk.get("size_bytes", 0),
                })
            if cool_local_images.get("merged"):
                cool_local_image_urls["merged"] = {
                    "url": f"/api/benchmark/{task.task_id}/cool-image?name=mosaic_merged.jpg",
                    "size_bytes": cool_local_images["merged"].get("size_bytes", 0),
                    "download_url": f"/api/benchmark/{task.task_id}/cool-image?name=mosaic_merged.jpg&download=1",
                }
            if cool_local_images.get("merged_preview"):
                cool_local_image_urls["merged_preview"] = {
                    "url": f"/api/benchmark/{task.task_id}/cool-image?name=mosaic_merged_preview.jpg",
                    "size_bytes": cool_local_images["merged_preview"].get("size_bytes", 0),
                }

        elif ip and pipeline_images:
            import urllib.parse
            for chunk in pipeline_images.get("chunks", []):
                image_urls["chunks"].append({
                    "name": chunk["name"],
                    "url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(chunk['path'], safe='')}",
                    "size_bytes": chunk.get("size_bytes", 0),
                })
            if pipeline_images.get("merged"):
                image_urls["merged"] = {
                    "url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(pipeline_images['merged']['path'], safe='')}",
                    "size_bytes": pipeline_images["merged"].get("size_bytes", 0),
                    "download_url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(pipeline_images['merged']['path'], safe='')}&download=1",
                }
            if pipeline_images.get("merged_preview"):
                image_urls["merged_preview"] = {
                    "url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(pipeline_images['merged_preview']['path'], safe='')}",
                    "size_bytes": pipeline_images["merged_preview"].get("size_bytes", 0),
                }

        # Build COOL image URLs (from saved COOL images or local download)
        cool_image_urls = {"chunks": [], "merged": None, "merged_preview": None}
        if cool_pipeline_images and not local_images:
            import urllib.parse
            for chunk in cool_pipeline_images.get("chunks", []):
                cool_image_urls["chunks"].append({
                    "name": chunk["name"],
                    "url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(chunk['path'], safe='')}",
                    "size_bytes": chunk.get("size_bytes", 0),
                })
            if cool_pipeline_images.get("merged"):
                cool_image_urls["merged"] = {
                    "url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(cool_pipeline_images['merged']['path'], safe='')}",
                    "size_bytes": cool_pipeline_images["merged"].get("size_bytes", 0),
                }
            if cool_pipeline_images.get("merged_preview"):
                cool_image_urls["merged_preview"] = {
                    "url": f"/api/benchmark/{task.task_id}/image?path={urllib.parse.quote(cool_pipeline_images['merged_preview']['path'], safe='')}",
                    "size_bytes": cool_pipeline_images["merged_preview"].get("size_bytes", 0),
                }

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "mode": getattr(task, "mode", "diy"),
            "current_run": getattr(task, "current_run", "cool"),
            "chunk_size": task.chunk_size,
            "instance_type": task.instance_type,
            "instance_id": getattr(task, "instance_id", None),
            "instance_ip": getattr(task, "instance_ip", None),
            "ssh_user": getattr(task, "ssh_user", "ec2-user"),
            "key_path": os.environ.get("EC2_KEY_PATH", "key.pem"),
            "build_progress": task.build_progress,
            "pipeline_stage": getattr(self, "_last_pipeline_stage", ""),
            "pipeline_pct": getattr(self, "_last_pipeline_pct", 0),
            "pipeline_images": image_urls,
            "cool_pipeline_images": cool_image_urls if not cool_local_images else cool_local_image_urls,
            "results": task.results,
            "error": task.error,
            "elapsed_time": __import__("time").time() - task.start_time,
        }

    async def cleanup(self):
        await self.instance_manager.cleanup_orphaned_instances() if hasattr(
            self.instance_manager, "cleanup_orphaned_instances"
        ) else None


async def create_app():
    orchestrator = PipelineOrchestrator()
    await orchestrator.initialize()

    app = web.Application()
    app["orchestrator"] = orchestrator

    # CORS middleware at module level (aiohttp 3.14 requires this)

    async def handle_start_benchmark(request):
        data = await request.json()
        task_id = await orchestrator.start_benchmark(
            instance_type=data.get("instanceType", "m7g.xlarge"),
            max_instances=data.get("maxInstances", 1),
            chunk_size=data.get("chunkSize", 18),
            opencv_version="5",
            mode=data.get("mode", "diy"),
        )
        resp = web.json_response({"taskId": task_id})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    async def handle_benchmark_status(request):
        task_id = request.match_info["task_id"]
        status = await orchestrator.get_benchmark_status(task_id)
        resp = web.json_response(status)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    async def handle_cleanup(request):
        await orchestrator.cleanup()
        resp = web.json_response({"message": "Cleanup complete"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    async def handle_config(request):
        resp = web.json_response({
            "ami_id": orchestrator.marketplace_ami_id,
            "diy_ami_id": orchestrator.diy_ami_id,
            "key_path": os.environ.get("EC2_KEY_PATH", "key.pem"),
        })
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    async def handle_instance_logs(request):
        task_id = request.match_info["task_id"]
        task = orchestrator.tasks.get(task_id)
        if not task or not getattr(task, "instance_ip", None):
            resp = web.json_response({"error": "No instance found"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        ip = task.instance_ip
        ssh_user = getattr(task, "ssh_user", "ec2-user")
        import subprocess
        key_path = os.environ.get("EC2_KEY_PATH", "key.pem")
        try:
            proc = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 "-i", key_path, f"{ssh_user}@{ip}",
                 "sudo tail -50 /tmp/pipeline-mcp.log 2>/dev/null || echo 'No log file'"],
                capture_output=True, text=True, timeout=15
            )
            logs = proc.stdout if proc.returncode == 0 else proc.stderr
            resp = web.json_response({"logs": logs, "ip": ip})
        except Exception as e:
            resp = web.json_response({"error": str(e), "ip": ip}, status=500)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    async def handle_terminate_instance(request):
        task_id = request.match_info["task_id"]
        task = orchestrator.tasks.get(task_id)
        if not task or not getattr(task, "instance_id", None):
            resp = web.json_response({"error": "No instance found"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        instance_id = task.instance_id
        try:
            await orchestrator.instance_manager.terminate_instance(instance_id)
            resp = web.json_response({"message": f"Instance {instance_id} terminated"})
        except Exception as e:
            resp = web.json_response({"error": str(e)}, status=500)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    async def handle_instance_image(request):
        task_id = request.match_info["task_id"]
        task = orchestrator.tasks.get(task_id)
        if not task or not getattr(task, "instance_ip", None):
            resp = web.json_response({"error": "No instance"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        ip = task.instance_ip
        ssh_user = getattr(task, "ssh_user", "ec2-user")
        import urllib.parse
        img_path = urllib.parse.unquote(request.query.get("path", ""))
        if not img_path:
            resp = web.json_response({"error": "No path"}, status=400)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        key_path = os.environ.get("EC2_KEY_PATH", "key.pem")
        # Use SCP to fetch the image
        import subprocess, tempfile, os as _os
        tmp = tempfile.mktemp(suffix=".jpg")
        try:
            proc = subprocess.run(
                ["scp", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                 f"{ssh_user}@{ip}:{img_path}", tmp],
                capture_output=True, text=True, timeout=60
            )
            if proc.returncode != 0 or not _os.path.exists(tmp):
                raise Exception(proc.stderr[-200:])
            resp = web.FileResponse(tmp)
            if request.query.get("download"):
                resp.headers['Content-Disposition'] = 'attachment; filename="mosaic_merged.jpg"'
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        except Exception as e:
            resp = web.json_response({"error": str(e)}, status=500)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    app.router.add_post("/api/benchmark/run", handle_start_benchmark)
    app.router.add_get("/api/benchmark/{task_id}/status", handle_benchmark_status)
    app.router.add_get("/api/config", handle_config)
    app.router.add_get("/api/benchmark/{task_id}/logs", handle_instance_logs)
    app.router.add_post("/api/benchmark/{task_id}/terminate", handle_terminate_instance)
    app.router.add_get("/api/benchmark/{task_id}/image", handle_instance_image)
    async def handle_cool_image(request):
        task_id = request.match_info["task_id"]
        task = orchestrator.tasks.get(task_id)
        if not task or not getattr(task, "cool_local_images", None):
            resp = web.json_response({"error": "No cool local images"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        name = request.query.get("name", "")
        import tempfile, os as _os
        local_dir = os.path.join(tempfile.gettempdir(), f"pipeline-results-{task_id}-cool")
        file_path = os.path.join(local_dir, name)
        if not _os.path.exists(file_path):
            resp = web.json_response({"error": "File not found"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        resp = web.FileResponse(file_path)
        if request.query.get("download"):
            resp.headers['Content-Disposition'] = 'attachment; filename="mosaic_merged.jpg"'
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    app.router.add_get("/api/benchmark/{task_id}/cool-image", handle_cool_image)
    async def handle_local_image(request):
        task_id = request.match_info["task_id"]
        task = orchestrator.tasks.get(task_id)
        if not task or not getattr(task, "local_images", None):
            resp = web.json_response({"error": "No local images"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        name = request.query.get("name", "")
        import tempfile
        local_dir = os.path.join(tempfile.gettempdir(), f"pipeline-results-{task_id}")
        file_path = os.path.join(local_dir, name)
        if not os.path.exists(file_path):
            resp = web.json_response({"error": "File not found"}, status=404)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        resp = web.FileResponse(file_path)
        if request.query.get("download"):
            resp.headers['Content-Disposition'] = 'attachment; filename="mosaic_merged.jpg"'
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    app.router.add_get("/api/benchmark/{task_id}/local-image", handle_local_image)
    app.router.add_post("/api/instances/cleanup", handle_cleanup)
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}, headers={"Access-Control-Allow-Origin": "*"}))

    # Serve frontend
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    if os.path.exists(frontend_path):
        async def index_handler(request):
            resp = web.FileResponse(os.path.join(frontend_path, "index.html"))
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            return resp
        app.router.add_get("/", index_handler)
        app.router.add_static("/", path=frontend_path, name="static")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logger.info("Pipeline orchestrator running on port 8080")
    logger.info("Frontend at http://localhost:8080/ (served by orchestrator)")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(create_app())