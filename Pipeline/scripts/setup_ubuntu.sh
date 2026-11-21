#!/usr/bin/env bash
# One-time setup of the GlassesData offline pipeline on Ubuntu 24.04 with an NVIDIA GPU
# (written for an RTX 5060 laptop GPU: Blackwell, sm_120, 8 GB).
#
# Idempotent: run it again at any time. Every step checks first and only does what is missing.
# Each command is printed before it runs, and every step that needs sudo asks first.
#
#   Pipeline/scripts/setup_ubuntu.sh                  interactive (recommended the first time)
#   Pipeline/scripts/setup_ubuntu.sh --yes --models   no questions, also download the model weights
#
# Options:
#   -y, --yes        answer yes to every question
#   --cuda VARIANT   PyTorch wheel index: auto (default: from the driver), cu130 or cu128
#   --models         download the model weights now (Depth Anything V2 metric indoor small/base/large,
#                    SAM 2.1 hiera-large, MediaPipe hand landmarker; about 2.8 GB into local/cache)
#   --no-models      do not offer the download (the backends fetch what they need on first use)
#   --no-sam2        skip SAM 2 (objects.mask_backend = "sam2"; a git install)
#   --with-wilor     also clone and install WiLoR (hands2d.backend = "wilor"; needs MANO, see below)
#   --with-hamer     also clone and install HaMeR (hands2d.backend = "hamer"; needs MANO, see below)
#   -h, --help       show this help
#
# Environment: GLASSESDATA_PIPELINE_VENV (default local/venv-pipeline in the repository),
#              GLASSESDATA_PYTHON (default python3.12, Ubuntu 24.04's Python),
#              GLASSESDATA_SRC (default local/src: the MuJoCo Menagerie, WiLoR and HaMeR clones),
#              GLASSESDATA_SAM2_REF, GLASSESDATA_MENAGERIE_REF, GLASSESDATA_WILOR_REF, GLASSESDATA_HAMER_REF
#              (the git commit installed; default: the commits pinned below, checked in November 2026;
#              a branch or tag name works too, e.g. GLASSESDATA_MENAGERIE_REF=main).
#
# The project keeps nothing outside its own folder: the venv, the clones, pip's cache, the model
# weights and temporary files all go into the repository's gitignored local/ folder (cache and
# temp variables: scripts/local_dirs.sh). Nothing tracked by git is changed. Written on a Mac and
# checked with bash -n only: the first run on the Linux laptop is its test (then run
# scripts/omen_smoke.sh).

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd -- "$PIPELINE_DIR/.." && pwd)
readonly SCRIPT_DIR PIPELINE_DIR REPO_DIR
if [[ $EUID -eq 0 ]]; then  # before anything is created in local/
    printf 'Run this as your own user, not with sudo: it asks for sudo itself where needed (local/ belongs to you).\n' >&2
    exit 1
fi
# shellcheck source=local_dirs.sh
source "$SCRIPT_DIR/local_dirs.sh"  # caches and TMPDIR into local/ (exported for every command below)
VENV=${GLASSESDATA_PIPELINE_VENV:-$LOCAL_DIR/venv-pipeline}
PYTHON=${GLASSESDATA_PYTHON:-python3.12}
SRC=${GLASSESDATA_SRC:-$LOCAL_DIR/src}
VENV_PY="$VENV/bin/python"
MENAGERIE="$SRC/mujoco_menagerie"
readonly VENV PYTHON SRC VENV_PY MENAGERIE
readonly TORCH_INDEX=https://download.pytorch.org/whl
# Sources installed from git, pinned to commits (main of each repository on 2026-11-21).
SAM2_REF=${GLASSESDATA_SAM2_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}            # 2024-12-16
MENAGERIE_REF=${GLASSESDATA_MENAGERIE_REF:-370c43965323c0944b433b202aac5058b5f907a9}  # 2026-11-10
WILOR_REF=${GLASSESDATA_WILOR_REF:-fcb911312a38fa8badd30d9656a167485d61b8f9}          # 2026-04-07
HAMER_REF=${GLASSESDATA_HAMER_REF:-3a01849f4148352e9260b69bf28b65d1671a4905}          # 2026-02-07
readonly SAM2_REF MENAGERIE_REF WILOR_REF HAMER_REF
readonly SAM2_GIT="git+https://github.com/facebookresearch/sam2.git@$SAM2_REF"
# Model weights (Hugging Face hub ids; the backends use the same ids, see backends/depth_anything.py
# and configs/objects.toml [objects.sam2]).
readonly HF_MODELS=(
    depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
    depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf
    depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
    facebook/sam2.1-hiera-large
)

ASSUME_YES=0
CUDA_VARIANT=auto
MODELS=ask
NO_SAM2=0
WITH_WILOR=0
WITH_HAMER=0
WARNINGS=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -y | --yes) ASSUME_YES=1 ;;
        --cuda)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            CUDA_VARIANT=$2
            shift
            ;;
        --cuda=*) CUDA_VARIANT=${1#--cuda=} ;;
        --models) MODELS=yes ;;
        --no-models) MODELS=no ;;
        --no-sam2) NO_SAM2=1 ;;
        --with-wilor) WITH_WILOR=1 ;;
        --with-hamer) WITH_HAMER=1 ;;
        -h | --help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
case $CUDA_VARIANT in
    auto | cu130 | cu128) ;;
    *) printf 'Unknown --cuda %s (auto, cu130 or cu128)\n' "$CUDA_VARIANT" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ output and questions

if [[ -t 1 ]]; then
    BOLD=$'\033[1m' DIM=$'\033[2m' GREEN=$'\033[32m' YELLOW=$'\033[33m' RED=$'\033[31m' RESET=$'\033[0m'
else
    BOLD="" DIM="" GREEN="" YELLOW="" RED="" RESET=""
fi
readonly BOLD DIM GREEN YELLOW RED RESET

step() { printf '\n%s== %s%s\n' "$BOLD" "$*" "$RESET"; }
info() { printf '   %s\n' "$*"; }
ok() { printf '   %sok%s    %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '   %swarn%s  %s\n' "$YELLOW" "$RESET" "$*"; WARNINGS=$((WARNINGS + 1)); }
die() { printf '\n%sstopped:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# ask "question": 0 for yes. Without a terminal (and without --yes) the answer is no.
ask() {
    local reply
    if [[ $ASSUME_YES -eq 1 ]]; then
        info "$1 [y/N] y (--yes)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        info "$1 [y/N] n (no terminal to ask on)"
        return 1
    fi
    read -r -p "   $1 [y/N] " reply || return 1
    [[ $reply == [Yy]* ]]
}

# Print a command (on stderr, so it shows even when the output is redirected), then run it.
run() {
    printf '   %s$ %s%s\n' "$DIM" "$*" "$RESET" >&2
    "$@"
}

run_sudo() {
    run sudo "$@"
}

is_installed() {
    local status
    # shellcheck disable=SC2016  # ${Status} is dpkg-query's field, not a shell variable
    status=$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null) || return 1
    [[ $status == "install ok installed" ]]
}

pip_install() {
    run "$VENV_PY" -m pip install --disable-pip-version-check --prefer-binary "$@"
}

# opencv-python (pulled by ultralytics, HaMeR, ...) and opencv-contrib-python both install cv2/:
# keep only the contrib build (cv2.aruco for eval-charuco).
fix_opencv() {
    if "$VENV_PY" -m pip show --disable-pip-version-check opencv-python >/dev/null 2>&1; then
        warn "opencv-python is installed next to opencv-contrib-python (two copies of cv2; cv2.aruco may vanish)"
        if ask "Remove opencv-python and reinstall opencv-contrib-python?"; then
            run "$VENV_PY" -m pip uninstall -y opencv-python || true
            pip_install --force-reinstall --no-deps "opencv-contrib-python>=4.10" || warn "reinstall failed"
        fi
    fi
}

# The installed torch/torchvision as pip constraints, so that no later install replaces the CUDA
# build with another one (e.g. a package asking for a newer torch would pull PyPI's build).
CONSTRAINTS=$(mktemp)
# shellcheck disable=SC2064  # expand now: the path is fixed
trap "rm -f -- '$CONSTRAINTS'" EXIT
write_constraints() {
    "$VENV_PY" - "$CONSTRAINTS" <<'EOF'
import sys
from importlib import metadata
lines = []
for name in ("torch", "torchvision"):
    try:
        lines.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
EOF
}

# ------------------------------------------------------------------ 1. this computer and the GPU

DRIVER_MAJOR=0
check_system() {
    step "This computer"
    local name="unknown" id="" version=""
    if [[ -r /etc/os-release ]]; then
        # shellcheck source=/dev/null
        name=$(. /etc/os-release && printf '%s' "${PRETTY_NAME:-unknown}")
        # shellcheck source=/dev/null
        id=$(. /etc/os-release && printf '%s' "${ID:-}")
        # shellcheck source=/dev/null
        version=$(. /etc/os-release && printf '%s' "${VERSION_ID:-}")
    fi
    if [[ $id == ubuntu && $version == 24.04 ]]; then
        ok "$name"
    else
        warn "written for Ubuntu 24.04; this is $name"
        ask "Continue anyway?" || exit 1
    fi
    [[ $(uname -m) == x86_64 ]] && ok "x86_64, kernel $(uname -r)" || warn "$(uname -m): only x86_64 wheels were checked"
    info "pipeline: $PIPELINE_DIR"
    info "venv:     $VENV"
    info "clones:   $SRC"
    info "caches:   $XDG_CACHE_HOME (Hugging Face $HF_HOME, pip $PIP_CACHE_DIR)"
    info "temp:     $TMPDIR"
    local free_gb
    free_gb=$(df -Pk "$REPO_DIR" | awk 'NR == 2 {printf "%d", $4 / 1048576}')
    if [[ $free_gb -lt 25 ]]; then
        warn "only ${free_gb} GB free in $REPO_DIR: the venv (torch with CUDA, ~8 GB) and the models (~3 GB) need about 15 GB"
    else
        ok "${free_gb} GB free in $REPO_DIR"
    fi
}

check_gpu() {
    step "NVIDIA driver"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        local line driver
        line=$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -n 1)
        driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
        DRIVER_MAJOR=${driver%%.*}
        [[ $DRIVER_MAJOR =~ ^[0-9]+$ ]] || DRIVER_MAJOR=0
        ok "$line"
        if [[ $DRIVER_MAJOR -ge 570 ]]; then
            ok "driver $driver supports Blackwell (sm_120 needs 570 or newer; CUDA 13 wheels need 580 or newer)"
            return
        fi
        warn "driver $driver is too old for an RTX 50-series GPU (needs 570 or newer)"
    else
        warn "nvidia-smi does not work: no NVIDIA driver is loaded"
        if command -v mokutil >/dev/null 2>&1 && mokutil --sb-state 2>/dev/null | grep -q enabled; then
            info "Secure Boot is on: a driver installed by ubuntu-drivers is signed by Ubuntu and loads as is."
        fi
    fi
    if command -v ubuntu-drivers >/dev/null 2>&1; then
        info "Ubuntu's recommended driver for this GPU:"
        ubuntu-drivers devices 2>/dev/null | sed -n '/recommended/p' | sed 's/^/        /' || true
    fi
    if ask "Install the recommended NVIDIA driver with ubuntu-drivers (sudo; reboot afterwards)?"; then
        run_sudo ubuntu-drivers install || die "ubuntu-drivers failed (see above)"
        die "driver installed: reboot, then run this script again"
    fi
    die "the pipeline's learned models and MuJoCo Warp need the NVIDIA driver (570 or newer for this GPU)"
}

# ------------------------------------------------------------------ 2. Ubuntu packages

install_packages() {
    step "Ubuntu packages"
    # python3.12-venv/-dev: the venv; build-essential: wheels that compile (chumpy, SAM 2 without
    # its CUDA kernels); ffmpeg: Rerun's video decoding and inspecting recordings; git: the clones;
    # libegl1 + libgl1: MuJoCo's EGL renderer (MUJOCO_GL=egl) and OpenCV; libglib2.0-0t64: OpenCV;
    # libosmesa6: MuJoCo's software renderer fallback (MUJOCO_GL=osmesa).
    local wanted=(python3.12-venv python3.12-dev build-essential ffmpeg git libegl1 libgl1 libglib2.0-0t64 libosmesa6)
    local missing=() package
    for package in "${wanted[@]}"; do
        is_installed "$package" || missing+=("$package")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        ok "${wanted[*]} installed"
        return
    fi
    info "to install: ${missing[*]}"
    ask "Install them with apt (sudo)?" || die "the pipeline needs these packages"
    run_sudo apt-get update
    if ! run_sudo apt-get install -y "${missing[@]}"; then
        die "apt could not install them. If it says \"Unable to locate package\", enable universe: sudo add-apt-repository universe"
    fi
    ok "installed ${missing[*]}"
}

# ------------------------------------------------------------------ 3. the Python environment

python_version() {
    "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

setup_venv() {
    step "Python environment"
    local system_version venv_version
    command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found (Ubuntu 24.04 has python3.12; set GLASSESDATA_PYTHON)"
    system_version=$(python_version "$PYTHON") || die "$PYTHON does not run"
    if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        die "$PYTHON is Python $system_version; the pipeline needs 3.11 or newer"
    fi
    ok "$PYTHON is Python $system_version"
    if [[ -x $VENV_PY ]]; then
        if venv_version=$(python_version "$VENV_PY") && [[ $venv_version == "$system_version" ]]; then
            ok "venv $VENV (Python $venv_version)"
        else
            warn "the venv at $VENV uses Python ${venv_version:-?}, not $system_version (or is broken)"
            if ask "Delete it and create it again?"; then
                run rm -rf -- "$VENV"
            fi
        fi
    fi
    if [[ ! -x $VENV_PY ]]; then
        run mkdir -p -- "$(dirname -- "$VENV")"
        run "$PYTHON" -m venv "$VENV"
        ok "created $VENV"
    fi
    pip_install --upgrade pip setuptools wheel
}

torch_status() {
    # Prints "version|cuda|arches|available" of the venv's torch, or nothing.
    "$VENV_PY" - <<'EOF' 2>/dev/null || true
import torch
arches = ",".join(torch.cuda.get_arch_list()) if torch.version.cuda else ""
print(f"{torch.__version__}|{torch.version.cuda}|{arches}|{torch.cuda.is_available()}")
EOF
}

install_torch() {
    step "PyTorch with CUDA"
    local variant=$CUDA_VARIANT status
    if [[ $variant == auto ]]; then
        # CUDA 13.0 wheels (the current PyTorch releases) need driver 580+; the CUDA 12.8 index
        # (the oldest that supports Blackwell) ends at torch 2.11 and needs 570+.
        if [[ $DRIVER_MAJOR -ge 580 ]]; then variant=cu130; else variant=cu128; fi
        info "driver $DRIVER_MAJOR -> $variant ($TORCH_INDEX/$variant)"
    fi
    status=$(torch_status)
    if [[ -n $status ]]; then
        IFS='|' read -r version cuda arches available <<<"$status"
        if [[ $arches == *sm_120* && $available == True ]]; then
            ok "torch $version (CUDA $cuda, $arches) sees the GPU"
            return
        fi
        warn "torch $version (CUDA ${cuda:-none}; arches ${arches:-none}; GPU available: $available) cannot run on this GPU"
        ask "Replace it with the $variant build?" || die "the learned backends need a CUDA build of torch with sm_120"
        pip_install --force-reinstall torch torchvision --index-url "$TORCH_INDEX/$variant" || die "the torch install failed"
    else
        pip_install torch torchvision --index-url "$TORCH_INDEX/$variant" || die "the torch install failed"
    fi
    status=$(torch_status)
    [[ -n $status ]] || die "torch does not import after the install (see above)"
    IFS='|' read -r version cuda arches available <<<"$status"
    ok "torch $version (CUDA $cuda)"
    [[ $arches == *sm_120* ]] || warn "torch.cuda.get_arch_list() has no sm_120: $arches"
    [[ $available == True ]] || warn "torch.cuda.is_available() is False (driver? reboot after a driver install?)"
}

install_requirements() {
    step "Pipeline requirements ($PIPELINE_DIR/requirements.txt)"
    write_constraints
    info "keeping $(tr '\n' ' ' <"$CONSTRAINTS")"
    pip_install -c "$CONSTRAINTS" -r "$PIPELINE_DIR/requirements.txt" || die "pip could not install requirements.txt (see above)"
    fix_opencv
    if [[ $NO_SAM2 -eq 1 ]]; then
        info "SAM 2 skipped (--no-sam2): objects.mask_backend = \"sam2\" reports the missing package"
    elif "$VENV_PY" -c 'import sam2' >/dev/null 2>&1; then
        ok "SAM 2 installed (reinstall another commit: $VENV_PY -m pip uninstall sam2, then run this again)"
    else
        # Official package from the repository. SAM2_BUILD_CUDA=0: no nvcc needed (the optional
        # CUDA kernel only fills small holes in masks); --no-build-isolation: build against the
        # installed torch instead of downloading another one.
        info "SAM 2 (objects.mask_backend = \"sam2\"): pip install from $SAM2_GIT"
        if ! run env SAM2_BUILD_CUDA=0 "$VENV_PY" -m pip install --disable-pip-version-check --no-build-isolation \
            -c "$CONSTRAINTS" "$SAM2_GIT"; then
            warn "SAM 2 did not install; objects.mask_backend = \"gt\" still works on synthetic recordings"
        fi
    fi
    local names=(numpy cv2 av mediapipe torch transformers open3d trimesh coacd mujoco mujoco_warp warp mink rsl_rl
                 rerun huggingface_hub)
    [[ $NO_SAM2 -eq 1 ]] || names+=(sam2)
    run "$VENV_PY" - "${names[@]}" <<'EOF' || warn "some packages do not import (see above)"
import importlib
import sys
bad = []
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
        print(f"   {name:<16} {getattr(module, '__version__', '')}")
    except Exception as error:  # noqa: BLE001
        bad.append(name)
        print(f"   {name:<16} NOT IMPORTABLE: {type(error).__name__}: {error}")
import cv2
assert hasattr(cv2, "aruco"), "cv2.aruco missing (opencv-contrib-python)"
raise SystemExit(1 if bad else 0)
EOF
}

# ------------------------------------------------------------------ git sources at a pinned commit

# clone_at URL DIR REF [--recursive]: a shallow checkout of REF (commit, branch or tag) in DIR;
# an existing clone at another commit is moved to REF after asking.
clone_at() {
    local url=$1 dir=$2 ref=$3 recursive=${4:-}
    if [[ ! -d $dir/.git ]]; then
        if [[ -e $dir ]]; then
            warn "$dir exists but is not a git clone; move it away and run this again"
            return 1
        fi
        run mkdir -p -- "$dir" || return 1
        run git -C "$dir" init -q || return 1
        run git -C "$dir" remote add origin "$url" || return 1
    else
        local head
        head=$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo none)
        if [[ $head == "$ref" ]]; then
            ok "$dir at $ref"
            [[ -z $recursive ]] || run git -C "$dir" submodule update --init --recursive --depth 1 || return 1
            return 0
        fi
        info "$dir is at $head, the pinned commit is $ref"
        ask "Check out $ref in $dir (local changes there are kept by git or stop the checkout)?" || return 0
    fi
    run git -C "$dir" fetch -q --depth 1 origin "$ref" || return 1
    run git -C "$dir" checkout -q FETCH_HEAD || return 1
    if [[ -n $recursive ]]; then
        run git -C "$dir" submodule update --init --recursive --depth 1 || return 1
    fi
    ok "$dir at $(git -C "$dir" rev-parse --short HEAD) ($ref)"
}

# ------------------------------------------------------------------ 4. MuJoCo Menagerie

clone_menagerie() {
    step "MuJoCo Menagerie (robot models: $MENAGERIE at $MENAGERIE_REF)"
    if [[ ! -d $MENAGERIE/.git ]] && ! ask "Clone the Menagerie into $MENAGERIE (shallow, ~300 MB)?"; then
        warn "no Menagerie: retarget, physics and train fail"
        return
    fi
    clone_at https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE" "$MENAGERIE_REF" \
        || { warn "the Menagerie checkout failed (see above)"; return; }
    [[ -f $MENAGERIE/shadow_hand/right_hand.xml ]] || warn "shadow_hand/right_hand.xml missing in $MENAGERIE"
}

# ------------------------------------------------------------------ 5. model weights

download_models() {
    step "Model weights (Hugging Face cache: $HF_HOME)"
    if [[ $MODELS == no ]]; then
        info "skipped (--no-models): each backend downloads what it uses on first use"
        return
    fi
    info "Depth Anything V2 metric indoor small/base/large (~0.1/0.4/1.3 GB), SAM 2.1 hiera-large (~0.9 GB),"
    info "MediaPipe hand_landmarker.task (8 MB, into $LOCAL_DIR/cache/)"
    if [[ $MODELS == ask ]] && ! ask "Download them now (about 2.8 GB)?"; then
        info "skipped: each backend downloads what it uses on first use"
        return
    fi
    run "$VENV_PY" - "${HF_MODELS[@]}" <<'EOF' || warn "some downloads failed (see above); the backends retry on first use"
import sys
from huggingface_hub import snapshot_download
failed = 0
for repo in sys.argv[1:]:
    try:
        path = snapshot_download(repo_id=repo)
        print(f"   ok    {repo} -> {path}")
    except Exception as error:  # noqa: BLE001
        failed += 1
        print(f"   FAIL  {repo}: {type(error).__name__}: {error}")
raise SystemExit(1 if failed else 0)
EOF
    (cd -- "$PIPELINE_DIR" && run "$VENV_PY" -c \
        'from glassesdata_pipeline.backends.mediapipe_hands import resolve_model; print("   ok   ", resolve_model())') \
        || warn "the MediaPipe hand model did not download (hands2d retries on first use)"
}

# ------------------------------------------------------------------ 6. optional: WiLoR / HaMeR

explain_mano() {
    # $1 = where MANO_RIGHT.pkl must be
    local target=$1
    if [[ -f $target ]]; then
        ok "MANO: $target"
        return
    fi
    warn "MANO_RIGHT.pkl is missing: $target"
    info "MANO is licensed separately and cannot be downloaded by a script:"
    info "  1. register at https://mano.is.tue.mpg.de (Sign Up) and accept the license;"
    info "  2. Download > \"Models & Code\" (mano_v1_2.zip), unzip it;"
    info "  3. copy mano_v1_2/models/MANO_RIGHT.pkl to $target"
    info "The MANO pickle needs the chumpy package to load. chumpy's last release predates Python 3.11"
    info "and NumPy 1.24: if 'import chumpy' fails, see the WiLoR/HaMeR issue trackers for the patched"
    info "fork they recommend, or convert MANO_RIGHT.pkl to plain numpy arrays once in a Python 3.10 env."
}

install_wilor() {
    step "WiLoR (hands2d.backend = \"wilor\", optional)"
    local repo="$SRC/WiLoR"
    clone_at https://github.com/rolpotamias/WiLoR.git "$repo" "$WILOR_REF" --recursive \
        || { warn "the WiLoR checkout failed (see above)"; return; }
    write_constraints
    # WiLoR's requirements.txt targets Python 3.10 + torch 2.0 (conda). Into this venv: keep our
    # torch (constraints) and opencv-contrib-python (drop opencv-python), skip the gradio demo UI.
    local filtered
    filtered=$(mktemp)
    grep -vE '^(opencv-python|gradio)([<=> ]|$)' "$repo/requirements.txt" >"$filtered" || true
    info "installing WiLoR's requirements without opencv-python and gradio:"
    sed 's/^/        /' "$filtered"
    pip_install -c "$CONSTRAINTS" -r "$filtered" || warn "some of WiLoR's requirements failed (see above)"
    rm -f -- "$filtered"
    fix_opencv
    local file
    for file in detector.pt wilor_final.ckpt; do
        if [[ -f $repo/pretrained_models/$file ]]; then
            ok "pretrained_models/$file"
        elif ask "Download pretrained_models/$file from the WiLoR Hugging Face space?"; then
            run "$VENV_PY" - "$repo" "$file" <<'EOF' || warn "download of $file failed"
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, name = Path(sys.argv[1]), sys.argv[2]
src = hf_hub_download(repo_id="rolpotamias/WiLoR", repo_type="space", filename=f"pretrained_models/{name}")
(repo / "pretrained_models").mkdir(exist_ok=True)
shutil.copy(src, repo / "pretrained_models" / name)
print("   ok   ", repo / "pretrained_models" / name)
EOF
        fi
    done
    explain_mano "$repo/mano_data/MANO_RIGHT.pkl"
    (cd -- "$repo" && run "$VENV_PY" -c 'import wilor.models, ultralytics; print("   wilor imports")') \
        || warn "WiLoR does not import yet (see above)"
    info "use it: --set hands2d.backend=wilor --set hands2d.repo_dir=$repo"
}

install_hamer() {
    step "HaMeR (hands2d.backend = \"hamer\", optional)"
    local repo="$SRC/hamer"
    clone_at https://github.com/geopavlakos/hamer.git "$repo" "$HAMER_REF" --recursive \
        || { warn "the HaMeR checkout failed (see above)"; return; }
    write_constraints
    # The backend's default hand boxes come from MediaPipe (hands2d.hamer_detector = "mediapipe"),
    # so ViTPose / detectron2 (HaMeR's own demo detector) are not installed here.
    (cd -- "$repo" && pip_install -c "$CONSTRAINTS" -e .) || warn "pip install -e $repo failed (see above)"
    fix_opencv
    if [[ -d $repo/_DATA/hamer_ckpts ]]; then
        ok "$repo/_DATA (checkpoints)"
    elif ask "Download HaMeR's checkpoints (bash fetch_demo_data.sh, a few GB, into $repo/_DATA)?"; then
        (cd -- "$repo" && run bash fetch_demo_data.sh) || warn "fetch_demo_data.sh failed"
    fi
    explain_mano "$repo/_DATA/data/mano/MANO_RIGHT.pkl"
    (cd -- "$repo" && run "$VENV_PY" -c 'import hamer.models; print("   hamer imports")') \
        || warn "HaMeR does not import yet (see above)"
    info "use it: --set hands2d.backend=hamer --set hands2d.repo_dir=$repo"
}

# ------------------------------------------------------------------ 7. final check

final_check() {
    step "Check: GPU, torch, MuJoCo (EGL), Warp"
    run nvidia-smi || warn "nvidia-smi failed"
    (cd -- "$PIPELINE_DIR" && run env MUJOCO_GL=egl "$VENV_PY" - <<'EOF') || warn "the check failed (see above)"
import sys
import torch
print(f"   torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
print(f"   arch list {torch.cuda.get_arch_list()}")
ok = torch.cuda.is_available() and "sm_120" in torch.cuda.get_arch_list()
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    x = torch.randn(1024, 1024, device="cuda")
    print(f"   GPU {name}, capability sm_{cap[0]}{cap[1]}, matmul ok: {bool(torch.isfinite(x @ x).all())}")
else:
    print("   torch.cuda.is_available() is False")
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
    with mujoco.Renderer(model, 64, 64) as renderer:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        renderer.render()
    print(f"   mujoco {mujoco.__version__}: EGL rendering ok")
except Exception as error:  # noqa: BLE001
    ok = False
    print(f"   mujoco EGL rendering failed: {type(error).__name__}: {error}")
try:
    import warp as wp
    wp.init()
    print(f"   warp {wp.config.version}: CUDA devices {[d.alias for d in wp.get_cuda_devices()]}")
except Exception as error:  # noqa: BLE001
    ok = False
    print(f"   warp failed: {type(error).__name__}: {error}")
sys.exit(0 if ok else 1)
EOF
}

check_system
check_gpu
install_packages
setup_venv
install_torch
install_requirements
clone_menagerie
download_models
[[ $WITH_WILOR -eq 1 ]] && install_wilor
[[ $WITH_HAMER -eq 1 ]] && install_hamer
if [[ $WITH_WILOR -eq 0 && $WITH_HAMER -eq 0 ]]; then
    info ""
    info "WiLoR / HaMeR (optional hand backends) are not installed: --with-wilor / --with-hamer (MANO needed)."
fi
final_check

step "Done"
if [[ $WARNINGS -gt 0 ]]; then
    info "$WARNINGS warning(s) above."
fi
info "MuJoCo renders off-screen with EGL on the NVIDIA GPU; set this in every shell that runs the pipeline"
info "(omen_smoke.sh sets it itself):"
info "  export MUJOCO_GL=egl"
if [[ $MENAGERIE != "$LOCAL_DIR/src/mujoco_menagerie" ]]; then
    info "  export MUJOCO_MENAGERIE=$MENAGERIE     # retarget.menagerie_path defaults to local/src/mujoco_menagerie"
fi
info "First run (tests, synthetic end-to-end, real models, RL; log folder + summary):"
info "  \"$SCRIPT_DIR/omen_smoke.sh\" [a real recording folder]"
