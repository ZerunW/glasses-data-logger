# shellcheck shell=bash
# Sourced by setup_ubuntu.sh and omen_smoke.sh (needs REPO_DIR, the repository root).
#
# The project keeps nothing outside its own folder: every download, library cache and temporary
# file goes into the repository's gitignored local/ folder. The same variables as
# glassesdata_pipeline/paths.py (environment()), which sets them again when the package is
# imported; tests/test_paths.py checks that the two lists match. A variable that is already set wins.

LOCAL_DIR="$REPO_DIR/local"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$LOCAL_DIR/cache}
export HF_HOME=${HF_HOME:-$LOCAL_DIR/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$LOCAL_DIR/cache/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$LOCAL_DIR/cache/matplotlib}
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-$LOCAL_DIR/cache/warp}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$LOCAL_DIR/cache/nvidia/ComputeCache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_DIR/cache/torchinductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$LOCAL_DIR/cache/triton}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-$LOCAL_DIR/cache/ultralytics}
export FVCORE_CACHE=${FVCORE_CACHE:-$LOCAL_DIR/cache/iopath}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$LOCAL_DIR/cache/pip}
export TMPDIR=${TMPDIR:-$LOCAL_DIR/tmp}
mkdir -p -- "$TMPDIR"
