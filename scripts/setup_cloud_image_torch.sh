#!/usr/bin/env bash
set -euo pipefail

if [ -d ".venv" ]; then
  echo "Existing .venv detected. Move it aside first, for example:"
  echo "  mv .venv .venv-old"
  exit 1
fi

python - <<'PY'
import torch

print("Base torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("The cloud image Python does not expose a CUDA PyTorch runtime.")

device_name = torch.cuda.get_device_name(0)
arch_list = torch.cuda.get_arch_list()
print("GPU:", device_name)
print("Supported CUDA arches:", arch_list)

if "5090" in device_name and "sm_120" not in arch_list:
    raise SystemExit("The cloud image PyTorch does not support RTX 5090 sm_120.")
PY

uv venv --python "$(command -v python)" --system-site-packages
source .venv/bin/activate

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "gymnasium[atari,accept-rom-license]==0.28.1" \
  "ale-py==0.8.1" \
  "numpy==1.21.6" \
  "opencv-python-headless==4.7.0.72" \
  "moviepy==1.0.3" \
  "imageio==2.28.1" \
  "imageio-ffmpeg==0.3.0" \
  "pygame==2.1.0" \
  "setuptools==67.7.2" \
  "tensorboard==2.11.2" \
  "cloudpickle==2.2.1" \
  "pandas==1.3.5" \
  "matplotlib==3.5.3" \
  "huggingface-hub==0.11.1"

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps \
  "stable-baselines3==2.0.0a1"

uv run --no-sync python - <<'PY'
import cv2
import moviepy
import stable_baselines3
import torch

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA arches:", torch.cuda.get_arch_list())
print("OpenCV:", cv2.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("MoviePy:", moviepy.__version__)
PY
