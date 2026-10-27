#!/usr/bin/env bash
# Install the inference/customization runtime into the selected Python 3.10 env.
# Never creates an environment, installs system packages, or downloads model weights.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

case "${1:-}" in
  -h|--help)
    echo 'Usage: PYTHON=/path/to/env/bin/python bash setup.sh [--check]'
    echo 'Default: install into the selected Python 3.10 environment, then verify.'
    echo '--check: verify the existing environment only (requires a >=70 GB GPU).'
    exit 0 ;;
  --check)
    exec "$PYTHON" "$ROOT/scripts/check_environment.py" ;;
  '') ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo 'Too many arguments' >&2; exit 2; fi

"$PYTHON" - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 10) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('This recipe requires Linux x86_64 and Python 3.10.')
print('Installing into:', sys.executable, flush=True)
PY
for source in \
  "$ROOT/depth_prediction/thirdparty/Depth-Anything-3/src/depth_anything_3/api.py" \
  "$ROOT/depth_prediction/thirdparty/MoGe/moge/model/v1.py"; do
  if [[ ! -f "$source" ]]; then
    echo 'Missing source submodules. Run: git submodule update --init --recursive' >&2
    exit 1
  fi
done
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo 'CUDA 12.6 Toolkit not found; set CUDA_HOME to its installation directory.' >&2
  exit 1
fi
if ! "$CUDA_HOME/bin/nvcc" --version | grep -q 'release 12.6'; then
  echo 'This recipe requires CUDA Toolkit 12.6.' >&2; exit 1
fi
for tool in git c++ ffmpeg; do
  if ! command -v "$tool" >/dev/null; then
    echo "Missing system prerequisite: $tool (install it before setup)." >&2; exit 1
  fi
done
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"

"$PYTHON" -m pip install setuptools==83.0.0 wheel==0.47.0 packaging==26.3 ninja==1.13.2
"$PYTHON" -m pip install -c "$ROOT/constraints.txt" \
  torch==2.8.0 torchvision==0.23.0 xformers==0.0.32.post2 \
  --index-url https://download.pytorch.org/whl/cu126
"$PYTHON" -m pip install -c "$ROOT/constraints.txt" -r "$ROOT/requirements.txt"
"$PYTHON" -m pip install --no-deps \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
"$PYTHON" -m pip install --no-deps --no-build-isolation \
  'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae' \
  'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@219fc3e3250fc6d7b314fee3045e32e582903fc5' \
  'utils3d-moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37'

# Wrappers load pinned submodule sources directly. Installing their entire
# pyprojects would add unrelated demos/training and conflicting NumPy constraints.
"$PYTHON" "$ROOT/scripts/check_environment.py"