#!/usr/bin/env bash
# ============================================================================
# Setup script for a fresh Vast AI pod (Ubuntu + NVIDIA CUDA base image).
#
# IMPORTANT: env.yml in the repo pins PyTorch 1.2.0 / CUDA 10.0 / Python 3.7.
# That combination CANNOT run on modern GPUs (RTX 30/40-series, A100, etc.)
# because their compute capability requires CUDA 11.8+/12.x driver support.
# This script installs a MODERN stack instead (system python3 + PyTorch
# cu121) on the assumption that this is what was actually used to produce
# the RTX 4090 results in the paper. The code itself uses no version-specific
# APIs (plain torch.distributed + torch.multiprocessing.spawn), so this is
# expected to be a safe substitution. If something fails, the 10-epoch test
# script (test_single_run.sh) is exactly where you'll find out early.
#
# Uses whatever python3 the base image ships (e.g. 3.12 on Ubuntu 24.04
# "noble" images) rather than pinning 3.11, since some Vast AI base images
# don't carry python3.11 in their default apt repos.
#
# Run this once per pod, as the pod's default user (root is fine on Vast AI).
# ============================================================================
set -euo pipefail

REPO_URL="https://github.com/khuethuc/HCMUT-Specialized-Project.git"
REPO_DIR="$HOME/repo"
DATA_DIR="$HOME/data"
VENV_DIR="$HOME/ngc_env"

echo "[1/6] Installing system packages..."
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git wget curl ca-certificates

echo "[2/6] Creating virtualenv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "[3/6] Installing PyTorch (CUDA 12.1 build) + deps..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install \
    numpy pandas scikit-learn pillow matplotlib scipy tqdm requests \
    torchsummary quadprog datasets huggingface_hub

echo "[4/6] Cloning repo..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo already present at $REPO_DIR, pulling latest..."
    git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
# Sanity check: the pod scripts rely on --noise-profile, which must be
# committed+pushed to the repo before this clone will contain it.
if ! grep -q "noise-profile" "$REPO_DIR/trainer.py"; then
    echo "  [WARN] --noise-profile not found in cloned trainer.py."
    echo "         Make sure you committed and pushed the dataloader.py/"
    echo "         trainer.py changes before renting pods, or rsync your"
    echo "         local working copy here instead of relying on git clone."
fi

echo "[5/6] Downloading datasets (HAM10000 + Fed-ISIC2019)..."
mkdir -p "$DATA_DIR"
cd "$REPO_DIR"
python dataset/download_ham10000.py --out "$DATA_DIR/ham10000"
python dataset/download_fedisic2019.py --out-dir "$DATA_DIR/fedisic2019"
# If Fed-ISIC2019 fails with a HuggingFace auth/gating error, run:
#   huggingface-cli login
# then re-run the download_fedisic2019.py line above.

echo "[6/6] Done. Activate with: source $VENV_DIR/bin/activate"
echo "Repo at:  $REPO_DIR"
echo "Data at:  $DATA_DIR"
