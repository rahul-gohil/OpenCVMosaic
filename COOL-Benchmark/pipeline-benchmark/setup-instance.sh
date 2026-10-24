#!/bin/bash
# Setup script for COOL AMI instances
# Run this via user-data or manually on a launched instance.
# Assumes COOL AMI (Graviton5) with /opt/cool already installed.

set -e

echo "=== Setting up OpenCVMosaic pipeline on COOL AMI ==="

# 1. Clone the repo
REPO_DIR=/opt/OpenCVMosaic
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/YOUR_REPO/OpenCVMosaic.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# 2. Use COOL's Python environment
source /opt/cool/venvs/python_3.12/bin/activate

# 3. Install pipeline dependencies (numpy, scipy are already in COOL venv)
pip install --quiet numpy scipy 2>/dev/null || true

# 4. Deploy pipeline runner
cp pipeline-benchmark/pipeline-runner.py /opt/pipeline-runner.py
chmod +x /opt/pipeline-runner.py

# 5. Verify
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} from {cv2.__file__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"

echo "=== Setup complete ==="
echo "Pipeline runner at /opt/pipeline-runner.py"
echo "Repo at $REPO_DIR"