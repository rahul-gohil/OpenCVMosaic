# OpenCVMosaic benchmark on AWS Graviton5 (COOL vs pip OpenCV)

Harness for running the **OpenCVMosaic** orthomosaic pipeline on real AWS
Graviton5 (Neoverse-V3) instances and comparing two OpenCV builds **on the same
machine, same OS and same Python version**:

| Mode | OpenCV | Python | OS / instance |
|---|---|---|---|
| **DIY** | `opencv-python-headless==5.0.0.93` (PyPI wheel) | 3.12 (dedicated venv) | COOL AMI, m9g/c9g |
| **COOL** | COOL OpenCV 5.1.0-dev (Marketplace AMI) | 3.12 (COOL venv) | COOL AMI, m9g/c9g |

**Compare** runs DIY first, reconfigures the instance to COOL, then runs COOL,
so both results come from one instance and identical hardware.

> **What this workload measures.** The mosaic pipeline is dominated by SIFT,
> descriptor matching, RANSAC homography and warping. COOL accelerates
> `resize`, `cvtColor`, adaptive thresholding and contour detection via
> KleidiCV — operations the mosaic barely uses. Measured end-to-end results on
> this pipeline are therefore **parity** (≈55 s both). The harness is the right
> tool to quantify that honestly; it is not a demonstration of a COOL speedup.

---

## Architecture

```
 your machine                                  AWS us-east-1
┌──────────────────────────────────┐          ┌──────────────────────────────────────────┐
│ pipeline-orchestrator.py :8080   │  HTTPS   │ EC2 m9g/c9g (COOL Graviton5 AMI)         │
│  ├─ serves frontend/index.html   │ ───────► │  └─ pipeline-mcp-server.py :8080         │
│  ├─ launches/terminates instances│  SSH/SCP │       ├─ /opt/cool venv  (COOL mode)     │
│  ├─ polls /pipeline-status       │ ◄─────── │       ├─ /opt/diy-venv    (DIY mode)      │
│  └─ downloads results            │          │       └─ runs: main.py run-chunks, merge │
└──────────────────────────────────┘          └──────────────────────────────────────────┘
```

* The instance clones this repository and copies the MCP server during
  cloud-init.
* The instance holds the 86-image dataset in-repo (`assets/`), so no S3 bucket
  or AWS credentials are needed on the instance.
* Images and metrics are streamed back to the orchestrator; heavy mosaics are
  fetched **on demand** (see *Outputs*).

---

## Prerequisites

1. **AWS account** with permission to launch/terminate EC2 instances and manage
   a security group. An 8 vCPU quota is enough for the default `m9g.xlarge`.
2. **EC2 key pair** in the target region (e.g. `mosaic-key`).
3. **COOL Marketplace subscription** — the AMI is required for *all* modes:
   <https://aws.amazon.com/marketplace/pp/prodview-igdweaoo6prb6>
   (7-day free trial; the AMI runs on `m9g`/`c9g` only).
4. **Python 3.10+** locally (the orchestrator).
5. AWS credentials via a named profile, environment variables or SSO.
   **Never commit keys.**

---

## Setup

```bash
git clone https://github.com/rahul-gohil/OpenCVMosaic.git
cd OpenCVMosaic/COOL-Benchmark

python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt          # aiohttp + boto3
```

### Configure

```bash
cp .env.example .env
cp config-marketplace.example.json config-marketplace.json
```

Edit `.env`:

```ini
AWS_DEFAULT_REGION=us-east-1
EC2_KEY_PAIR_NAME=mosaic-key
EC2_KEY_PATH=/home/you/.ssh/mosaic-key.pem
MARKETPLACE_AMI_ID=ami-xxxxxxxxxxxxxxxxx
```

These keys should live in your `.env` (git-ignored):
`AWS_DEFAULT_REGION`, `EC2_KEY_PAIR_NAME`, `EC2_KEY_PATH` and
`MARKETPLACE_AMI_ID`. Do not put AWS secret keys there — use a profile.

Edit `config-marketplace.json` with the AMI ID from your Marketplace
subscription page (region `us-east-1`). The orchestrator also accepts the AMI
from the UI / `MARKETPLACE_AMI_ID`.

AWS credentials are resolved by boto3 in the usual way:

```bash
export AWS_PROFILE=my-profile
# or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (shell only, never in the repo)
```

---

## Run

```bash
python3 pipeline-benchmark/pipeline-orchestrator.py
```

Open <http://localhost:8080/>, choose a mode, instance type and chunk size
(default 18), press **Run Benchmark**.

The orchestrator loads `.env` automatically. If you prefer, you can also use
`set -a; source .env; set +a` before starting it.

### Modes

* **DIY only** — launch COOL AMI → reconfigure to a Python 3.12 venv with pip
  OpenCV → run once.
* **COOL only** — launch COOL AMI → run once with COOL OpenCV.
* **Compare** — launch COOL AMI → reconfigure to DIY → run → reconfigure back to
  COOL → run → show both results side by side.

Both reconfigure paths are isolated: separate venvs/work dirs, `PYTHONPATH`
cleared for DIY, and the pip-wheel `ld.so.conf` entry removed before returning
to COOL.

---

## Measured Metrics

Per chunk (aggregated over all chunks, exact timings):

| Metric | Meaning |
|---|---|
| Load | decode/read input frame |
| SIFT | `detectAndCompute` |
| Matching | descriptor matching + ratio test |
| Homography | RANSAC model estimation/acceptance |
| Stitching | warp + blend + canvas writes |
| Saving | write chunk canvas/preview |
| Frames | processed frames (includes chunk overlaps) |

Merge stage reports SIFT / Matching / Homography / Stitching / total merge time.
Results also include canvas size (Mpx), content %, and OpenCV version/path so you
can verify which build actually ran.

---

## Outputs

* **Generated strips & merged result** — COOL and DIY panels side by side
  (single panel in single-mode runs).
* **Results** — combined chunk + merge per-stage timings, canvas, OpenCV build,
  and a parity/speedup line.
* **Downloads**
  * Chunk strips and the merged preview are downloaded automatically
    (a few MB; ~10 s on a slow uplink).
  * The **full-resolution mosaic** (≈19 MB each) is fetched **on demand** from
    the instance the first time you click *Download full mosaic*, then cached
    locally. Set `DOWNLOAD_FULL_CHUNKS=1` to also download full-size chunk JPEGs.
* **Local cache** — `/tmp/pipeline-results-<task_id>` and
  `/tmp/pipeline-results-<task_id>-diy`.

---

## Instance lifetime, cost and cleanup

* Instances are **not terminated automatically** after a run — they are left
  running so you can inspect logs and download the full mosaics.
* Use **Terminate Instance** (or **Stop**) in the UI, or:

  ```bash
  aws ec2 terminate-instances --region us-east-1 --instance-ids i-...
  ```

* Full-resolution mosaics that were never downloaded are **lost when the
  instance is terminated**. The UI warns about this, and the on-demand endpoint
  returns HTTP 410 afterwards.
* COOL Marketplace charges per instance-hour in addition to EC2. Remember to
  terminate.
* Instances are tagged `BenchmarkTaskId`, `Mode`, `AutoTerminate=true`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `InsufficientInstanceCapacity` | Handled automatically: the launcher retries every subnet/AZ and finally lets EC2 choose. If all AZs are full, retry later or use another instance size. |
| `VcpuLimitExceeded` | Your account vCPU quota is exhausted (often by an instance left running). Terminate stale instances or request a quota increase. |
| `Marketplace AMI not configured` | Set `MARKETPLACE_AMI_ID` or `config-marketplace.json`. |
| Frontend says "Task not found" | The orchestrator was restarted; tasks live in memory. Start a new run. |
| No preview images | The instance may be terminated, or the download is still running (status shows *Downloading results*). |
| `410` when downloading the full mosaic | The instance is gone and the file was never cached. |
| SSH/SCP failures | Check `EC2_KEY_PATH`, key pair name, and that ports 22/8080 are reachable (security group). |
| Reconfigure fails on `dpkg` lock | Cloud-init/reconfigure waits for and stops `unattended-upgrades`; retry on a fresh instance if it persists. |

---

---

## Repository layout (active files)

```
COOL-Benchmark/
├── pipeline-benchmark/
│   └── pipeline-orchestrator.py    # REST API + frontend host + EC2 orchestration
├── agentcore/
│   └── instance-manager.py         # launch (AZ failover), SG, terminate
├── opencv-ami/
│   └── pipeline-mcp-server.py      # runs on the instance
├── frontend/
│   └── index.html                  # single-page UI
├── assets/                         # 86 scaled UAVPistachio frames + CITATION
├── config-marketplace.example.json
├── .env.example
└── requirements.txt
```

`shared/`, `pipeline-benchmark/pipeline-benchmark.py`, `pipeline-runner.py`,
`setup-instance.sh`, `opencv-ami/opencv-mcp-server.py` and
`opencv-ami/install-opencv-optimized.sh` come from the upstream
COOL-Benchmark and are **not used** by this harness.

---

## Notes on the COOL comparison

* The COOL AMI is built for Graviton5 (`m9g`/`c9g`); other families are not
  offered in the UI.
* This COOL build reports `KleidiCV 0.7.0`; the pip wheel ships KleidiCV
  `26.03`, so the "KleidiCV is only in COOL" assumption is not true.
* COOL adds an SVE baseline build, but the mosaic hot path (SIFT / matching /
  RANSAC / warp) does not benefit measurably. On this pipeline expect parity;
  COOL-targeted wins appear on resize/colour-convert/threshold/contour-heavy
  workloads.

---

## License / attribution

Dataset and citation: see `assets/CITATION.md`. The harness builds on the
upstream [COOL-Benchmark](https://github.com/opencv/COOL-Benchmark) structure.
