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
    instance_id: Optional[str] = None
    instance_ip: Optional[str] = None
    build_progress: Optional[Dict[str, Any]] = None
    end_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineOrchestrator:
    def __init__(self, region: str = None, marketplace_ami_id: str = None, diy_ami_id: str = None):
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-1"))
        self.marketplace_ami_id = marketplace_ami_id or os.environ.get("MARKETPLACE_AMI_ID", "")
        # Standard Graviton AMI for DIY-only mode (no COOL subscription needed)
        self.diy_ami_id = diy_ami_id or os.environ.get("DIY_AMI_ID", "ami-06575ffbdddc0e7d0")
        self.instance_manager = InstanceManager(region=self.region)
        self.build_manager = BuildManager(region=self.region)
        self.tasks: Dict[str, PipelineBenchmarkTask] = {}
        self.default_region = self.region
        self._last_pipeline_images = {}
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
                logger.info(f"DIY mode: using standard Graviton AMI {ami_id}")
            elif mode == "cool":
                # COOL-only: use COOL marketplace AMI
                ami_id = self.marketplace_ami_id
                if not ami_id or ami_id.startswith("ami-to-be"):
                    raise ValueError("Marketplace AMI not configured. Set MARKETPLACE_AMI_ID or config-marketplace.json")
                user_data = self._get_user_data()
                logger.info(f"COOL mode: using marketplace AMI {ami_id}")
            else:
                # Compare mode: use COOL marketplace AMI (runs both)
                ami_id = self.marketplace_ami_id
                if not ami_id or ami_id.startswith("ami-to-be"):
                    raise ValueError("Marketplace AMI not configured. Set MARKETPLACE_AMI_ID or config-marketplace.json")
                user_data = self._get_user_data()
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
                # DIY-only: run once with standard OpenCV
                diy_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size
                )
                task.build_progress = {"current_step": "Completed", "progress_percent": 90}
                task.results = {
                    "diy": diy_metrics,
                    "mode": "diy",
                }
                task.status = TaskStatus.COMPLETED

            elif mode == "cool":
                # COOL-only: run once with COOL OpenCV
                cool_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size
                )
                task.build_progress = {"current_step": "Completed", "progress_percent": 90}
                task.results = {
                    "cool": cool_metrics,
                    "mode": "cool",
                }
                task.status = TaskStatus.COMPLETED

            else:
                # Compare mode: run both COOL and DIY
                # Run COOL (marketplace) build first
                cool_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size
                )

                # Reconfigure to DIY (pip)
                task.build_progress = {"current_step": "Reconfiguring to DIY", "progress_percent": 70}
                await self._reconfigure_to_pip(ip)

                # Run DIY build
                diy_metrics = await self._run_pipeline_on_instance(
                    ip, task.chunk_size
                )

                # Compare
                task.build_progress = {"current_step": "Comparing results", "progress_percent": 90}
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
                try:
                    self.instance_manager.add_instance_to_pool(instance_id, task.instance_type, "marketplace", "arm64")
                    self.instance_manager.release_instance(instance_id)
                except Exception:
                    try:
                        await self.instance_manager.terminate_instance(instance_id)
                    except Exception:
                        pass

    async def _run_pipeline_on_instance(self, ip: str, chunk_size: int) -> dict:
        import asyncio as _asyncio
        async with aiohttp.ClientSession() as session:
            # Start pipeline (returns immediately now)
            async with session.post(
                f"http://{ip}:8080/run-pipeline",
                json={"chunk_size": chunk_size},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status not in (200, 202):
                    text = await resp.text()
                    raise Exception(f"Failed to start pipeline: {text}")

            # Poll for completion
            while True:
                await _asyncio.sleep(5)
                async with session.get(
                    f"http://{ip}:8080/pipeline-status",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    data = await resp.json()
                    status = data.get("status", "unknown")
                    progress = data.get("progress", {})
                    self._last_pipeline_images = data.get("images", {})
                    self._last_pipeline_stage = progress.get("current_step", "")
                    self._last_pipeline_pct = progress.get("progress_percent", 0)

                    if status == "completed":
                        metrics = data.get("metrics", {})
                        metrics["per_stage_timings"] = metrics.get("per_stage_timings", {})
                        return metrics
                    elif status == "failed":
                        raise Exception(data.get("error", "Pipeline failed"))

    async def _reconfigure_to_pip(self, ip: str):
        pip_version = "5.0.0.93"
        commands = [
            "sudo systemctl stop pipeline-mcp 2>/dev/null || true",
            f"pip3 install --break-system-packages opencv-python-headless=={pip_version} numpy scipy",
            "sudo systemctl restart pipeline-mcp 2>/dev/null || true",
        ]
        # Execute via SSH
        import subprocess
        key_path = os.environ.get("EC2_KEY_PATH", "key.pem")
        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", key_path,
                   f"ec2-user@{ip}", " && ".join(commands)]
        subprocess.run(ssh_cmd, check=True, timeout=600)

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
set -e
# COOL AMI is Amazon Linux (Graviton5) - has Python and git pre-installed
cd /opt
if [ ! -d "/opt/OpenCVMosaic" ]; then
    git clone --depth 1 https://github.com/rahul-gohil/OpenCVMosaic.git /opt/OpenCVMosaic
fi
cd /opt/OpenCVMosaic
source /opt/cool/venvs/python_3.12/bin/activate
pip install --quiet numpy scipy
cp /opt/OpenCVMosaic/COOL-Benchmark/opencv-ami/pipeline-mcp-server.py /opt/pipeline-mcp-server.py
# Start the pipeline MCP server as a systemd service
cat > /etc/systemd/system/pipeline-mcp.service << 'EOF'
[Unit]
Description=Pipeline MCP Server
After=network.target

[Service]
Type=simple
User=ubuntu
Environment=PYTHON=/opt/cool/venvs/python_3.12/bin/python
Environment=PIPELINE_DIR=/opt/OpenCVMosaic
ExecStart=/opt/cool/venvs/python_3.12/bin/python /opt/pipeline-mcp-server.py
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
        # Build image URLs through orchestrator proxy
        image_urls = {"chunks": [], "merged": None}
        if ip and pipeline_images:
            for chunk in pipeline_images.get("chunks", []):
                image_urls["chunks"].append({
                    "name": chunk["name"],
                    "url": f"/api/benchmark/{task.task_id}/image?path={chunk['path']}",
                    "size_bytes": chunk.get("size_bytes", 0),
                })
            if pipeline_images.get("merged"):
                image_urls["merged"] = {
                    "url": f"/api/benchmark/{task.task_id}/image?path={pipeline_images['merged']['path']}",
                    "size_bytes": pipeline_images["merged"].get("size_bytes", 0),
                }

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "mode": getattr(task, "mode", "diy"),
            "chunk_size": task.chunk_size,
            "instance_type": task.instance_type,
            "instance_id": getattr(task, "instance_id", None),
            "instance_ip": getattr(task, "instance_ip", None),
            "build_progress": task.build_progress,
            "pipeline_stage": getattr(self, "_last_pipeline_stage", ""),
            "pipeline_pct": getattr(self, "_last_pipeline_pct", 0),
            "pipeline_images": image_urls,
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
            instance_type=data.get("instanceType", "m7g.4xlarge"),
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
        import subprocess
        key_path = os.environ.get("EC2_KEY_PATH", "key.pem")
        try:
            proc = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                 "-i", key_path, f"ec2-user@{ip}",
                 "sudo tail -50 /var/log/pipeline-mcp.log 2>/dev/null || echo 'No log file'"],
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
        img_path = request.query.get("path", "")
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
                 f"ec2-user@{ip}:{img_path}", tmp],
                capture_output=True, text=True, timeout=30
            )
            if proc.returncode != 0 or not _os.path.exists(tmp):
                raise Exception(proc.stderr[-200:])
            resp = web.FileResponse(tmp)
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
    app.router.add_post("/api/instances/cleanup", handle_cleanup)
    app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}, headers={"Access-Control-Allow-Origin": "*"}))

    # Serve frontend
    frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    if os.path.exists(frontend_path):
        async def index_handler(request):
            return web.FileResponse(os.path.join(frontend_path, "index.html"))
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