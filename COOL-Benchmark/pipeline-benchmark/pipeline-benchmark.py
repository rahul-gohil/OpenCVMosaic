#!/usr/bin/env python3
"""
Pipeline Benchmark Orchestrator

Launches two Graviton5 EC2 instances (one with COOL AMI, one with DIY pip
OpenCV), runs the orthomosaic pipeline on both, and compares timings.

Usage:
    python pipeline-benchmark.py \
        --s3-input s3://bucket/inputs/pistachio \
        --s3-output s3://bucket/benchmark_results \
        --chunk-size 18 \
        --instance-type m9g.4xlarge \
        --region us-east-1 \
        --key-pair your-key-pair \
        --cool-ami ami-xxxxxxxxxxxxxxxxx
"""

import argparse
import json
import os
import subprocess
import sys
import time
import boto3
from typing import Dict, Any, Optional


def get_instance_cost(instance_type: str) -> float:
    """Hourly EC2 cost for Graviton5 instances (us-east-1 on-demand, 2026)."""
    costs = {
        "m9g.medium": 0.0416, "m9g.large": 0.0832, "m9g.xlarge": 0.1664,
        "m9g.2xlarge": 0.3328, "m9g.4xlarge": 0.6656, "m9g.8xlarge": 1.3312,
        "m9g.12xlarge": 1.9968, "m9g.16xlarge": 2.6624,
        "c9g.medium": 0.0372, "c9g.large": 0.0744, "c9g.xlarge": 0.1488,
        "c9g.2xlarge": 0.2976, "c9g.4xlarge": 0.5952, "c9g.8xlarge": 1.1904,
        "c9g.12xlarge": 1.7856, "c9g.16xlarge": 2.3808,
    }
    return costs.get(instance_type, 0.10)


def get_cool_software_cost(instance_type: str) -> float:
    """Hourly COOL software fee from AWS Marketplace."""
    # From marketplace pricing page
    costs = {
        "m9g.medium": 0.002, "m9g.large": 0.005, "m9g.xlarge": 0.01,
        "m9g.2xlarge": 0.02, "m9g.4xlarge": 0.063, "m9g.8xlarge": 0.125,
        "m9g.12xlarge": 0.188, "m9g.16xlarge": 0.251,
        "c9g.medium": 0.002, "c9g.large": 0.004, "c9g.xlarge": 0.009,
        "c9g.2xlarge": 0.017, "c9g.4xlarge": 0.056, "c9g.8xlarge": 0.111,
        "c9g.12xlarge": 0.167, "c9g.16xlarge": 0.223,
    }
    return costs.get(instance_type, 0.01)


def launch_instance(ec2_client, instance_type: str, ami_id: str, key_pair: str,
                    user_data: str, tags: dict) -> str:
    """Launch an EC2 instance and return its ID."""
    response = ec2_client.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_pair,
        UserData=user_data,
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": k, "Value": v} for k, v in tags.items()],
        }],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    print(f"  Launched {instance_id} ({instance_type}, AMI {ami_id})")
    return instance_id


def wait_for_instance(ec2_client, instance_id: str) -> str:
    """Wait for instance to be running and return its public IP."""
    print(f"  Waiting for {instance_id} to be running...")
    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    ip = response["Reservations"][0]["Instances"][0].get("PublicIpAddress")
    print(f"  Instance {instance_id} running at {ip}")
    return ip


def terminate_instance(ec2_client, instance_id: str):
    """Terminate an EC2 instance."""
    ec2_client.terminate_instances(InstanceIds=[instance_id])
    print(f"  Terminated {instance_id}")


def run_pipeline_on_instance(ip: str, s3_input: str, s3_output: str,
                            chunk_size: int, work_dir: str = "/tmp/scratchpad") -> dict:
    """Run the pipeline on a remote instance via SSH and return metrics."""
    import json as _json
    cmd = (
        f"cd /opt/OpenCVMosaic && "
        f"python3 pipeline-benchmark/pipeline-runner.py "
        f"--s3-input {s3_input} "
        f"--s3-output {s3_output} "
        f"--chunk-size {chunk_size} "
        f"--work-dir {work_dir}"
    )
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-i", os.environ.get("EC2_KEY_PATH", "key.pem"),
        f"ubuntu@{ip}",
        cmd,
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return {"error": result.stderr[-500:]}
    try:
        return _json.loads(result.stdout.strip().split("\n")[-1])
    except Exception as e:
        print(f"  Parse error: {e}")
        print(f"  stdout: {result.stdout[-500:]}")
        return {"error": str(e), "stdout": result.stdout[-500:]}


def reconfigure_to_pip(ip: str, opencv_version: str = "5"):
    """Reconfigure a COOL instance to use pip (DIY) OpenCV."""
    pip_version = "5.0.0.93" if opencv_version == "5" else "4.12.0.88"
    commands = [
        "sudo systemctl stop opencv-mcp 2>/dev/null || true",
        "sudo apt-get update -y -qq",
        "sudo apt-get install -y -qq python3-pip",
        f"pip3 install --break-system-packages opencv-python-headless=={pip_version} numpy",
        "sudo systemctl restart opencv-mcp 2>/dev/null || true",
    ]
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-i", os.environ.get("EC2_KEY_PATH", "key.pem"),
        f"ubuntu@{ip}",
        " && ".join(commands),
    ]
    subprocess.run(ssh_cmd, check=True, timeout=600)


def run_benchmark(args) -> dict:
    """Run the full benchmark: COOL first, then DIY on the same instance."""
    ec2 = boto3.client("ec2", region_name=args.region)
    key_pair = args.key_pair
    instance_type = args.instance_type

    # User data: install our pipeline code on the instance
    user_data = f"""#!/bin/bash
set -e
apt-get update -y
apt-get install -y git python3-pip python3-venv awscli
cd /opt
git clone --depth 1 https://github.com/YOUR_REPO/OpenCVMosaic.git /opt/OpenCVMosaic
cd /opt/OpenCVMosaic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Install our pipeline as a package
pip install -e .
echo "Pipeline ready" > /tmp/pipeline_ready
"""

    tags = {
        "Name": "opencv-pipeline-benchmark",
        "Project": "OpenCVMosaic",
        "AutoTerminate": "true",
    }

    # Launch one Graviton5 instance with COOL AMI
    print(f"\n{'='*60}")
    print("Phase 1: Launch Graviton5 instance with COOL AMI")
    print(f"{'='*60}")
    instance_id = launch_instance(ec2, instance_type, args.cool_ami, key_pair, user_data, tags)
    ip = wait_for_instance(ec2, instance_id)

    # Wait for pipeline to be ready
    print("  Waiting for pipeline setup...")
    time.sleep(120)

    results = {}

    # --- Phase 2: COOL benchmark ---
    print(f"\n{'='*60}")
    print("Phase 2: Run pipeline with COOL (OpenCV 5.1.0)")
    print(f"{'='*60}")
    cool_output = f"{args.s3_output}/cool"
    cool_metrics = run_pipeline_on_instance(ip, args.s3_input, cool_output, args.chunk_size)
    results["cool"] = cool_metrics
    print(f"  COOL results: {cool_metrics}")

    # --- Phase 3: Reconfigure to DIY (pip) ---
    print(f"\n{'='*60}")
    print("Phase 3: Reconfigure to DIY OpenCV (pip)")
    print(f"{'='*60}")
    reconfigure_to_pip(ip, "5")
    print("  Reconfigured to pip OpenCV")

    # --- Phase 4: DIY benchmark ---
    print(f"\n{'='*60}")
    print("Phase 4: Run pipeline with DIY (OpenCV 5.0.0)")
    print(f"{'='*60}")
    diy_output = f"{args.s3_output}/diy"
    diy_metrics = run_pipeline_on_instance(ip, args.s3_input, diy_output, args.chunk_size)
    results["diy"] = diy_metrics
    print(f"  DIY results: {diy_metrics}")

    # --- Phase 5: Terminate ---
    print(f"\n{'='*60}")
    print("Phase 5: Terminate instance")
    print(f"{'='*60}")
    terminate_instance(ec2, instance_id)

    # --- Phase 6: Compare ---
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")

    cool_time = results["cool"].get("total_time_s", 0)
    diy_time = results["diy"].get("total_time_s", 0)

    if cool_time > 0 and diy_time > 0:
        speedup = diy_time / cool_time
        print(f"  COOL total time:  {cool_time:.1f}s")
        print(f"  DIY total time:   {diy_time:.1f}s")
        print(f"  Speedup (COOL):   {speedup:.2f}x")
        print(f"  Time saved:       {diy_time - cool_time:.1f}s")

    # Cost comparison
    cool_cost = get_instance_cost(instance_type) * (cool_time / 3600)
    diy_cost = get_instance_cost(instance_type) * (diy_time / 3600)
    cool_software = get_cool_software_cost(instance_type) * (cool_time / 3600)
    print(f"\n  COOL EC2 cost:    ${cool_cost:.4f}")
    print(f"  COOL software:    ${cool_software:.4f}")
    print(f"  COOL total:       ${cool_cost + cool_software:.4f}")
    print(f"  DIY total:        ${diy_cost:.4f}")
    print(f"  COOL premium:     ${cool_cost + cool_software - diy_cost:.4f}")

    # Save results
    results["instance_type"] = instance_type
    results["chunk_size"] = args.chunk_size
    results["comparison"] = {
        "speedup": round(speedup, 2) if cool_time > 0 and diy_time > 0 else None,
        "time_saved_s": round(diy_time - cool_time, 1) if cool_time > 0 and diy_time > 0 else None,
        "cool_total_cost": round(cool_cost + cool_software, 4),
        "diy_total_cost": round(diy_cost, 4),
    }

    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-input", required=True, help="S3 prefix with input images")
    parser.add_argument("--s3-output", required=True, help="S3 prefix for benchmark results")
    parser.add_argument("--chunk-size", type=int, default=18, help="frames per chunk")
    parser.add_argument("--instance-type", default="m9g.4xlarge", help="EC2 instance type")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--key-pair", required=True, help="EC2 key pair name")
    parser.add_argument("--cool-ami", required=True, help="COOL marketplace AMI ID")
    parser.add_argument("--key-path", default="key.pem", help="Path to SSH private key")
    args = parser.parse_args()

    os.environ["EC2_KEY_PATH"] = args.key_path

    results = run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())