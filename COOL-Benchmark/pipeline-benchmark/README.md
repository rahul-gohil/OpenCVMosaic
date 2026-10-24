# Pipeline Benchmark for OpenCVMosaic

Adapts the COOL-Benchmark harness to run the full orthomosaic pipeline
(stitch + merge) on EC2 Graviton5 instances, comparing DIY (pip) OpenCV
vs COOL (marketplace) OpenCV on the same instance.

## Files

- `pipeline-runner.py` — Runs on EC2. Downloads dataset from S3, runs chunk
  stitching + merge, uploads results, returns JSON metrics.
- `pipeline-benchmark.py` — Orchestrator. Launches one Graviton5 instance,
  runs COOL first, reconfigures to DIY, runs again, compares results.
- `setup-instance.sh` — User-data / setup script for EC2 instances.

## Usage

```bash
# 1. Upload your dataset to S3
aws s3 sync outputs/pistachio_0*/ s3://your-bucket/inputs/pistachio/

# 2. Run the benchmark
python pipeline-benchmark.py \
    --s3-input s3://your-bucket/inputs/pistachio \
    --s3-output s3://your-bucket/benchmark_results \
    --chunk-size 18 \
    --instance-type m9g.4xlarge \
    --region us-east-1 \
    --key-pair your-key-pair \
    --cool-ami ami-xxxxxxxxxxxxxxxxx \
    --key-path ~/.ssh/your-key.pem
```

## Metrics reported

- Per-stage timings (download, stitch, merge, upload)
- Total wall-clock time
- Canvas dimensions and content %
- Peak memory (RSS)
- Cost per orthomosaic (EC2 + COOL software fee)
- Speedup ratio (DIY time / COOL time)

## Architecture

```
Single Graviton5 instance (m9g.4xlarge)
    │
    ├─ Phase 1: COOL build (marketplace AMI, OpenCV 5.1.0)
    │   └─ Run pipeline → collect metrics
    │
    ├─ Reconfigure: pip install opencv-python-headless==5.0.0.93
    │
    └─ Phase 2: DIY build (pip OpenCV, OpenCV 5.0.0)
        └─ Run pipeline → collect metrics

Compare: Phase 1 vs Phase 2 (same hardware, different OpenCV)
```
