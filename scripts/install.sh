#!/bin/bash
# One environment for GAM and the benchmarks.
#
#   bash scripts/install.sh                    # GAM only (training, your own robot)
#   bash scripts/install.sh --libero-pro DIR   # + LIBERO-PRO checkout at DIR
#   bash scripts/install.sh --robotwin DIR     # + RoboTwin 2.0 checkout at DIR (sapien, mplib, curobo)
#
# Run inside the conda / venv you want to use (python 3.11). Set TORCH_INDEX for your CUDA, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130   (default; cu126 / cu128 also work)
# Building curobo needs nvcc of the same major CUDA version as torch (CUDA_HOME) and a C++17 compiler.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
LIBERO_PRO=""; ROBOTWIN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --libero-pro) LIBERO_PRO="$(cd "$2" && pwd)"; shift 2 ;;
    --robotwin)   ROBOTWIN="$(cd "$2" && pwd)"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

python -m pip --version > /dev/null 2>&1 || python -m ensurepip --upgrade        # a bare conda env may come without pip
echo "== torch"
python -m pip install "torch==2.13.*" "torchvision==0.28.*" --index-url "$TORCH_INDEX"
echo "== GAM + perception stack"
python -m pip install "setuptools<81"                 # sapien 3.0.0b1 still imports pkg_resources
python -m pip install -r "$ROOT/requirements.txt"
python -m pip install -e "$ROOT"
python -c "import GAM, sam3, wilddet3d, mdm; from GAM.model.policy import GAMPolicy; print('GAM ok')"

if [ -n "$LIBERO_PRO" ]; then
  echo "== LIBERO-PRO ($LIBERO_PRO)"
  python -m pip install "robosuite==1.4.0" "bddl==1.0.1" "gym==0.25.2" "mujoco==3.2.3" "PyOpenGL-accelerate" "easydict" "hydra-core" "thop" "cloudpickle" "future"   # mujoco 3.3+ breaks robosuite 1.4
  python -m pip install -e "$LIBERO_PRO" --no-deps
  python -c "import libero, robosuite; print('LIBERO-PRO ok')"
  echo "   set LIBERO_DATASETS=<LIBERO demos root> for precompute"
fi

if [ -n "$ROBOTWIN" ]; then
  echo "== RoboTwin ($ROBOTWIN)"
  if [ -d "$ROBOTWIN/.git" ] && ! git -C "$ROBOTWIN" rev-parse HEAD | grep -q '^6e75b8b'; then
    echo "   WARNING: RoboTwin is not at commit 6e75b8b (the release was evaluated with it; later commits removed script/eval_policy.py)"
  fi
  python -m pip install "sapien==3.0.0b1" "mplib==0.2.1" "transforms3d==0.4.2" "gymnasium==0.29.1" toppra numpy-quaternion \
      "warp-lang==1.12.0" "pyglet<2" msgpack msgpack-numpy websockets termcolor moviepy av open3d
  # RoboTwin's two one-line patches (from its own install script)
  SP=$(python -c "import sapien, os; print(os.path.dirname(sapien.__file__))")
  grep -q 'encoding="utf-8"' "$SP/wrapper/urdf_loader.py" || sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SP/wrapper/urdf_loader.py"
  grep -q 'encoding="utf-8"' "$SP/wrapper/urdf_loader.py" && echo "   patched sapien urdf_loader (utf-8 open)" || echo "   WARNING: sapien urdf_loader patch did not apply"
  MP=$(python -c "import mplib, os; print(os.path.dirname(mplib.__file__))")
  sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$MP/planner.py"
  grep -q "or collide or not within_joint_limit" "$MP/planner.py" && echo "   WARNING: mplib planner patch did not apply" || echo "   patched mplib planner (RoboTwin's own install patch: no collision check in IK)"
  # curobo v0.7.8 compiled against this torch (C++17 is required by the CUDA 12.8+ / 13 headers)
  if ! python -c "import curobo" 2>/dev/null; then
    CB="$ROBOTWIN/envs/curobo"
    [ -d "$CB/src/curobo/curobolib/cpp" ] && [ -n "$(ls "$CB/src/curobo/curobolib/cpp" 2>/dev/null)" ] \
      || git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CB"
    sed -i 's/"nvcc": \[/"cxx": ["-std=c++17"],\n    "nvcc": [\n        "-std=c++17",/' "$CB/setup.py"
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.8 SETUPTOOLS_SCM_PRETEND_VERSION=0.7.8
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0+PTX}"
    (cd "$CB" && python -m pip install -e . --no-build-isolation)
  fi
  python -c "import sapien, mplib, curobo, warp; print('RoboTwin deps ok')"
  echo "   set ROBOTWIN_ROOT=$ROBOTWIN; RoboTwin's assets must be downloaded as in its README; ffmpeg must be on PATH"
fi
echo "== done"
