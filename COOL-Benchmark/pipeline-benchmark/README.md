# Pipeline benchmark

The supported entry point is the orchestrator:

```bash
cd ..
python pipeline-benchmark/pipeline-orchestrator.py
# open http://localhost:8080/
```

See [../README.md](../README.md) for prerequisites, configuration, modes,
outputs, cost/cleanup and troubleshooting.

## Files in this directory

| File | Used? | Purpose |
|---|---|---|
| `pipeline-orchestrator.py` | **yes** | REST API, EC2 orchestration, reconfiguration, result download, serves the UI |
| `pipeline-benchmark.py` | no | upstream single-instance S3-based runner (legacy) |
| `pipeline-runner.py` | no | upstream on-instance runner (legacy) |
| `setup-instance.sh` | no | upstream cloud-init script (legacy) |

The current flow is **DIY first, then COOL**, both on the COOL Graviton5 AMI:

1. launch COOL AMI;
2. reconfigure to a Python 3.12 venv with `opencv-python-headless` (DIY);
3. run `main.py run-chunks` + `main.py merge` (work dir `/tmp/scratchpad_diy`);
4. reconfigure back to COOL's venv + OpenCV;
5. run again (work dir `/tmp/scratchpad_cool`);
6. compare per-stage timings on the same hardware.
