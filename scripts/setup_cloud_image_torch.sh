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
  "gymnasium[atari]>=1.2.0" \
  "ale-py>=0.11.0" \
  "numpy" \
  "opencv-python-headless" \
  "moviepy" \
  "imageio" \
  "imageio-ffmpeg" \
  "pygame" \
  "setuptools" \
  "tensorboard" \
  "cloudpickle" \
  "pandas" \
  "matplotlib" \
  "huggingface-hub"

uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps \
  "stable-baselines3>=2.7.0"

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
