#!/usr/bin/env bash
# First run of the GlassesData pipeline on the Linux GPU laptop (after scripts/setup_ubuntu.sh).
#
# One command: every stage writes its own log into a timestamped folder, a failure does not stop
# the next stages (a stage that needs a failed stage's output is skipped), and a summary table is
# printed and saved at the end. Everything (logs, recordings, caches, temporary files) goes into the
# repository's gitignored local/ folder; nothing tracked by git is changed.
#
#   Pipeline/scripts/omen_smoke.sh                        synthetic stages 1-6
#   Pipeline/scripts/omen_smoke.sh local/recordings/REC   + stage 7 on a copy of a real recording
#
# Stages:
#   1 environment report (versions, GPU, EGL)      5 retarget, physics, viz with the Rerun export
#   2 full pytest (heavy tests included, -rs)       6 train: MuJoCo CPU, MuJoCo Warp GPU, then --play
#   3 synth 8 s -> run (gt backends) -> eval        7 a real recording: mediapipe hands, Depth Anything V2,
#   4 the same clip with Depth Anything V2 depth      SAM 2 masks, every step up to viz
#     (hands stay gt) -> eval; then SAM 2 masks
#
# Options:
#   --out DIR        log folder (default local/smoke/<date>_<time> in the repository)
#   --seconds S      synthetic clip length (default 8)
#   --seed N         synthetic seed (default 3)
#   --skip-tests     skip stage 2
#   --skip-train     skip stage 6
#   -h, --help       show this help
#
# Environment: GLASSESDATA_PIPELINE_VENV (default local/venv-pipeline in the repository),
#              MUJOCO_MENAGERIE (default local/src/mujoco_menagerie; GLASSESDATA_SRC replaces
#              local/src), MUJOCO_GL (default egl); caches and TMPDIR: scripts/local_dirs.sh.
# Exit code: 0 when every stage passed, 1 otherwise. Written on a Mac and checked with bash -n only.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd -- "$PIPELINE_DIR/.." && pwd)
readonly SCRIPT_DIR PIPELINE_DIR REPO_DIR
# shellcheck source=local_dirs.sh
source "$SCRIPT_DIR/local_dirs.sh"  # caches and TMPDIR into local/ (exported for every stage)
VENV=${GLASSESDATA_PIPELINE_VENV:-$LOCAL_DIR/venv-pipeline}
PY="$VENV/bin/python"
readonly VENV PY
export MUJOCO_GL=${MUJOCO_GL:-egl}
export MUJOCO_MENAGERIE=${MUJOCO_MENAGERIE:-${GLASSESDATA_SRC:-$LOCAL_DIR/src}/mujoco_menagerie}
export PYTHONUNBUFFERED=1

OUT=""
SECONDS_CLIP=8
SEED=3
SKIP_TESTS=0
SKIP_TRAIN=0
REAL=""

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --out) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; OUT=$2; shift ;;
        --seconds) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; SECONDS_CLIP=$2; shift ;;
        --seed) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; SEED=$2; shift ;;
        --skip-tests) SKIP_TESTS=1 ;;
        --skip-train) SKIP_TRAIN=1 ;;
        -h | --help) usage; exit 0 ;;
        -*) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
        *)
            [[ -z $REAL ]] || { printf 'Only one recording, please\n' >&2; exit 2; }
            REAL=$1
            ;;
    esac
    shift
done

[[ -x $PY ]] || { printf 'No venv at %s: run %s/setup_ubuntu.sh first\n' "$VENV" "$SCRIPT_DIR" >&2; exit 1; }
if [[ -n $REAL && ! -f $REAL/video.mp4 ]]; then
    printf '%s is not a recording folder (no video.mp4)\n' "$REAL" >&2
    exit 2
fi
OUT=${OUT:-$LOCAL_DIR/smoke/$(date +%Y%m%d_%H%M%S)}
mkdir -p -- "$OUT" || exit 1
OUT=$(cd -- "$OUT" && pwd)
readonly OUT
GT="$OUT/rec_gt"        # synthetic recording, gt backends
DA="$OUT/rec_da"        # the same clip, Depth Anything V2 depth
SAM="$OUT/rec_sam2"     # rec_da with SAM 2 masks for the objects step
COPY="$OUT/real"        # copy of the real recording

if [[ -t 1 ]]; then
    BOLD=$'\033[1m' DIM=$'\033[2m' GREEN=$'\033[32m' YELLOW=$'\033[33m' RED=$'\033[31m' RESET=$'\033[0m'
else
    BOLD="" DIM="" GREEN="" YELLOW="" RED="" RESET=""
fi
readonly BOLD DIM GREEN YELLOW RED RESET

# ------------------------------------------------------------------ stage bookkeeping

NAMES=() STATUSES=() TIMES=() NOTES=()
FAILED=0

record() {  # name status seconds note
    NAMES+=("$1") STATUSES+=("$2") TIMES+=("$3") NOTES+=("$4")
    case $2 in
        FAIL) FAILED=$((FAILED + 1)); printf '   %sFAIL%s  %s (%s s): %s\n' "$RED" "$RESET" "$1" "$3" "$4" ;;
        WARN) printf '   %sWARN%s  %s (%s s): %s\n' "$YELLOW" "$RESET" "$1" "$3" "$4" ;;
        SKIP) printf '   %sSKIP%s  %s: %s\n' "$DIM" "$RESET" "$1" "$4" ;;
        *) printf '   %sok%s    %s (%s s) %s\n' "$GREEN" "$RESET" "$1" "$3" "$4" ;;
    esac
}

skip() {  # name reason
    record "$1" SKIP 0 "$2"
}

LAST_RC=0
LAST_TIME=0
# run_logged LOG command...: runs the command in the pipeline folder, output into LOG.
run_logged() {
    local log=$1 started
    shift
    {
        printf '$ %s\n' "$*"
        printf '# %s, cwd %s\n\n' "$(date -Is)" "$PIPELINE_DIR"
    } >"$log"
    printf '   %s$ %s%s\n' "$DIM" "$*" "$RESET"
    printf '   %slog %s%s\n' "$DIM" "$log" "$RESET"
    started=$SECONDS
    (cd -- "$PIPELINE_DIR" && "$@") >>"$log" 2>&1
    LAST_RC=$?
    LAST_TIME=$((SECONDS - started))
    printf '\n# exit %d after %d s\n' "$LAST_RC" "$LAST_TIME" >>"$log"
    if [[ $LAST_RC -ne 0 ]]; then
        printf '   %slast lines of %s:%s\n' "$DIM" "$log" "$RESET"
        tail -n 15 -- "$log" | sed 's/^/      /'
    fi
    return "$LAST_RC"
}

# stage NAME LOG command...: run_logged + record ok / FAIL with NOTE_FN's output as the note.
NOTE=""
stage() {
    local name=$1 log=$2
    shift 2
    printf '\n%s== %s%s\n' "$BOLD" "$name" "$RESET"
    NOTE=""
    if run_logged "$log" "$@"; then
        record "$name" ok "$LAST_TIME" "$NOTE"
        return 0
    fi
    record "$name" FAIL "$LAST_TIME" "exit $LAST_RC; see $(basename -- "$log")"
    return 1
}

pipeline() {  # python -m glassesdata_pipeline ...
    "$PY" -m glassesdata_pipeline "$@"
}

# A pipeline run stage: also lists the steps that did not run (failed / skipped / blocked).
run_stage() {
    local name=$1 log=$2 problems
    shift 2
    printf '\n%s== %s%s\n' "$BOLD" "$name" "$RESET"
    run_logged "$log" "$PY" -m glassesdata_pipeline run "$@"
    local rc=$LAST_RC
    problems=$(awk '$2 ~ /^(failed|skipped|blocked|unavailable|not-implemented)$/ {printf "%s%s=%s", sep, $1, $2; sep=", "}' "$log")
    if [[ $rc -ne 0 ]]; then
        record "$name" FAIL "$LAST_TIME" "exit $rc${problems:+; $problems}"
        return 1
    fi
    if [[ -n $problems ]]; then
        record "$name" WARN "$LAST_TIME" "$problems"
    else
        record "$name" ok "$LAST_TIME" "$(grep -m1 '^total' "$log" || true)"
    fi
    return 0
}

# Key numbers of derived/eval.json (synthetic recordings).
eval_note() {
    "$PY" - "$1" <<'EOF' 2>/dev/null || echo "(no eval.json)"
import json, sys
metrics = json.load(open(sys.argv[1] + "/derived/eval.json"))["metrics"]

def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out

m = {g: flat(v) for g, v in metrics.items() if isinstance(v, dict)}
rows = [("depth AbsRel", "depth", "pipeline_scaled.abs_rel", 1, "{:.3f}"),
        ("hand3d", "hand3d", "mpjpe", 1000, "{:.1f} mm"),
        ("scale err", "hand3d", "depth_scale.smooth_rel_abs.mean", 100, "{:.1f} %"),
        ("ATE", "slam", "ate_se3.rmse", 1000, "{:.1f} mm"),
        ("gravity", "imu", "gravity_error_deg.mean", 1, "{:.2f} deg"),
        ("world", "hands_world", "mpjpe", 1000, "{:.1f} mm"),
        ("elbow", "arms", "elbow.all.mean", 100, "{:.1f} cm"),
        ("obj0", "objects", "objects.obj0.translation_m.mean", 100, "{:.1f} cm")]
parts = []
for label, group, key, scale, fmt in rows:
    v = m.get(group, {}).get(key)
    if isinstance(v, (int, float)):
        parts.append(f"{label} " + fmt.format(v * scale))
print(", ".join(parts) or "(no metrics)")
EOF
}

# ------------------------------------------------------------------ 1. environment

printf '%sGlassesData pipeline smoke test%s -> %s\n' "$BOLD" "$RESET" "$OUT"
printf '\n%s== 1 environment%s\n' "$BOLD" "$RESET"
{
    printf '# %s\n' "$(date -Is)"
    printf 'host %s, %s\n' "$(hostname)" "$(uname -srm)"
    [[ -r /etc/os-release ]] && (. /etc/os-release && printf 'os %s\n' "${PRETTY_NAME:-?}")
    printf 'pipeline %s (git %s, %s changed files)\n' "$PIPELINE_DIR" \
        "$(git -C "$PIPELINE_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')" \
        "$(git -C "$PIPELINE_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    printf 'venv %s\nMUJOCO_GL=%s\nMUJOCO_MENAGERIE=%s (%s)\n' "$VENV" "$MUJOCO_GL" "$MUJOCO_MENAGERIE" \
        "$(git -C "$MUJOCO_MENAGERIE" log -1 --format='%h %cs' 2>/dev/null || echo 'missing')"
    printf 'caches %s (HF_HOME %s, WARP_CACHE_PATH %s), TMPDIR %s\n' "$XDG_CACHE_HOME" "$HF_HOME" "$WARP_CACHE_PATH" "$TMPDIR"
    printf 'ffmpeg: %s\n\n' "$(ffmpeg -version 2>/dev/null | head -n 1 || echo missing)"
    nvidia-smi 2>&1 || echo "nvidia-smi failed"
    echo
    "$PY" - <<'EOF' 2>&1
import importlib, platform
print(f"python {platform.python_version()}")
for name in ["numpy", "cv2", "av", "mediapipe", "torch", "torchvision", "transformers", "huggingface_hub", "open3d",
             "trimesh", "coacd", "mujoco", "mujoco_warp", "warp", "mink", "rsl_rl", "tensordict", "rerun", "sam2"]:
    try:
        module = importlib.import_module(name)
        print(f"{name:<16} {getattr(module, '__version__', '?')}")
    except Exception as error:  # noqa: BLE001
        print(f"{name:<16} MISSING ({type(error).__name__}: {error})")
try:
    import torch
    print(f"torch CUDA {torch.version.cuda}, available {torch.cuda.is_available()}, arch list {torch.cuda.get_arch_list()}")
    if torch.cuda.is_available():
        print(f"GPU {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}, "
              f"memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
except Exception as error:  # noqa: BLE001
    print(f"torch check failed: {error}")
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
    with mujoco.Renderer(model, 32, 32) as renderer:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        renderer.render()
    print("mujoco renderer ok")
except Exception as error:  # noqa: BLE001
    print(f"mujoco renderer FAILED: {type(error).__name__}: {error}")
try:
    import warp as wp
    wp.init()
    print(f"warp CUDA devices {[d.alias for d in wp.get_cuda_devices()]}")
except Exception as error:  # noqa: BLE001
    print(f"warp FAILED: {type(error).__name__}: {error}")
EOF
} >"$OUT/1_environment.txt" 2>&1
sed 's/^/   /' "$OUT/1_environment.txt" | grep -E 'GPU|torch CUDA|MISSING|FAILED|renderer|warp CUDA|^   (host|os|pipeline)' || true
if grep -qE 'MISSING|FAILED|available False' "$OUT/1_environment.txt"; then
    record "1 environment" WARN 0 "something is missing; see 1_environment.txt"
else
    record "1 environment" ok 0 "1_environment.txt"
fi

# ------------------------------------------------------------------ 2. tests

if [[ $SKIP_TESTS -eq 1 ]]; then
    skip "2 pytest" "--skip-tests"
else
    printf '\n%s== 2 pytest (heavy tests included)%s\n' "$BOLD" "$RESET"
    run_logged "$OUT/2_pytest.log" "$PY" -m pytest tests -q -p no:cacheprovider -rs
    result=$(grep -E '^[=]* *[0-9]+ (passed|failed)|^[0-9]+ (passed|failed)' "$OUT/2_pytest.log" | tail -n 1 | tr -d '=' | sed 's/^ *//')
    skipped=$(grep -c '^SKIPPED' "$OUT/2_pytest.log" || true)
    if [[ $LAST_RC -eq 0 ]]; then
        record "2 pytest" ok "$LAST_TIME" "${result:-?} (skip reasons listed: $skipped)"
    else
        record "2 pytest" FAIL "$LAST_TIME" "${result:-exit $LAST_RC}"
    fi
fi

# ------------------------------------------------------------------ 3. synthetic clip, gt backends

HAVE_GT=0
if stage "3a synth ${SECONDS_CLIP} s seed $SEED" "$OUT/3a_synth.log" \
    "$PY" -m glassesdata_pipeline synth "$GT" --seconds "$SECONDS_CLIP" --seed "$SEED" --overwrite; then
    HAVE_GT=1
fi
if [[ $HAVE_GT -eq 1 ]]; then
    run_stage "3b run (gt backends)" "$OUT/3b_run_gt.log" "$GT"
    if stage "3c eval (gt backends)" "$OUT/3c_eval_gt.log" "$PY" -m glassesdata_pipeline eval "$GT"; then
        NOTES[${#NOTES[@]} - 1]=$(eval_note "$GT")
        printf '      %s\n' "${NOTES[${#NOTES[@]} - 1]}"
    fi
else
    skip "3b run (gt backends)" "no synthetic recording"
    skip "3c eval (gt backends)" "no synthetic recording"
fi

# ------------------------------------------------------------------ 4. Depth Anything V2 (hands stay gt), SAM 2

if [[ $HAVE_GT -eq 1 ]]; then
    rm -rf -- "$DA"
    mkdir -p -- "$DA"
    (cd -- "$GT" && tar --exclude=./derived -cf - .) | (cd -- "$DA" && tar -xf -)
    DA_SETS=(--set depth.backend=depth_anything_v2)
    if run_stage "4a run (Depth Anything V2 depth)" "$OUT/4a_run_da.log" "$DA" "${DA_SETS[@]}"; then
        if stage "4b eval (Depth Anything V2 depth)" "$OUT/4b_eval_da.log" \
            "$PY" -m glassesdata_pipeline eval "$DA" "${DA_SETS[@]}"; then
            NOTES[${#NOTES[@]} - 1]=$(eval_note "$DA")
            printf '      %s\n' "${NOTES[${#NOTES[@]} - 1]}"
        fi
        rm -rf -- "$SAM"
        cp -a -- "$DA" "$SAM"
        SAM_SETS=("${DA_SETS[@]}" --set objects.mask_backend=sam2)
        if run_stage "4c objects (SAM 2 masks)" "$OUT/4c_objects_sam2.log" "$SAM" --steps objects "${SAM_SETS[@]}"; then
            if stage "4d eval (SAM 2 masks)" "$OUT/4d_eval_sam2.log" \
                "$PY" -m glassesdata_pipeline eval "$SAM" "${SAM_SETS[@]}"; then
                NOTES[${#NOTES[@]} - 1]=$(eval_note "$SAM")
                printf '      %s\n' "${NOTES[${#NOTES[@]} - 1]}"
            fi
        else
            skip "4d eval (SAM 2 masks)" "objects with SAM 2 failed"
        fi
    else
        skip "4b eval (Depth Anything V2 depth)" "the run failed"
        skip "4c objects (SAM 2 masks)" "the run failed"
        skip "4d eval (SAM 2 masks)" "the run failed"
    fi
else
    for name in "4a run (Depth Anything V2 depth)" "4b eval (Depth Anything V2 depth)" "4c objects (SAM 2 masks)" \
        "4d eval (SAM 2 masks)"; do
        skip "$name" "no synthetic recording"
    done
fi

# ------------------------------------------------------------------ 5. retarget, physics, viz + Rerun

if [[ $HAVE_GT -eq 1 ]]; then
    if run_stage "5a retarget, physics, viz (Rerun on)" "$OUT/5a_sim_viz.log" "$GT" \
        --steps retarget,physics,viz --set viz.rerun=on; then
        note=$("$PY" - "$GT" <<'EOF' 2>&1
import json, sys
from pathlib import Path
d = Path(sys.argv[1]) / "derived"
parts = []
for rel in ("retarget_shadow.npz", "physics.json", "viz/episode.rrd", "sim/replay.mp4"):
    p = d / rel
    parts.append(f"{rel} {p.stat().st_size / 1e6:.1f} MB" if p.exists() else f"{rel} MISSING")
try:
    phys = json.loads((d / "physics.json").read_text())
    parts.append(f"plausibility {phys.get('plausibility')}")
except Exception as error:  # noqa: BLE001
    parts.append(f"physics.json unreadable ({error})")
print(", ".join(parts))
EOF
)
        NOTES[${#NOTES[@]} - 1]="$note"
        printf '      %s\n' "$note"
        [[ $note == *MISSING* ]] && STATUSES[${#STATUSES[@]} - 1]=WARN
    fi
    stage "5b view --print" "$OUT/5b_view.log" "$PY" -m glassesdata_pipeline view "$GT" --print || true
else
    skip "5a retarget, physics, viz (Rerun on)" "no synthetic recording"
    skip "5b view --print" "no synthetic recording"
fi

# ------------------------------------------------------------------ 6. RL training

if [[ $SKIP_TRAIN -eq 1 ]]; then
    skip "6 train" "--skip-train"
elif [[ $HAVE_GT -eq 1 && -f $GT/derived/sim/scene.xml ]]; then
    trained=0
    stage "6a train MuJoCo CPU (16 envs, 3 it)" "$OUT/6a_train_mujoco.log" \
        "$PY" -m glassesdata_pipeline train "$GT" --backend mujoco --num-envs 16 --iterations 3 --run smoke_mujoco \
        && trained=1
    [[ $trained -eq 1 ]] && NOTES[${#NOTES[@]} - 1]=$(grep -m1 '^last checkpoint' "$OUT/6a_train_mujoco.log" || true)
    if stage "6b train MuJoCo Warp GPU (256 envs, 5 it)" "$OUT/6b_train_warp.log" \
        "$PY" -m glassesdata_pipeline train "$GT" --backend warp --num-envs 256 --iterations 5 --run smoke_warp; then
        trained=1
        NOTES[${#NOTES[@]} - 1]=$(grep -m1 '^last checkpoint' "$OUT/6b_train_warp.log" || true)
    fi
    if [[ $trained -eq 1 ]]; then
        stage "6c train --play (latest run)" "$OUT/6c_play.log" "$PY" -m glassesdata_pipeline train "$GT" --play || true
    else
        skip "6c train --play (latest run)" "no checkpoint"
    fi
else
    skip "6 train" "needs the synthetic recording's sim/scene.xml (stage 3)"
fi

# ------------------------------------------------------------------ 7. a real recording

if [[ -n $REAL ]]; then
    printf '\n%s== 7 real recording %s -> %s%s\n' "$BOLD" "$REAL" "$COPY" "$RESET"
    rm -rf -- "$COPY"
    mkdir -p -- "$COPY"
    (cd -- "$REAL" && tar --exclude=./derived -cf - .) | (cd -- "$COPY" && tar -xf -)
    REAL_SETS=(--set hands2d.backend=mediapipe --set depth.backend=depth_anything_v2 --set objects.mask_backend=sam2)
    run_stage "7a run real (mediapipe, DA V2, SAM 2)" "$OUT/7a_run_real.log" "$COPY" "${REAL_SETS[@]}" || true
    if [[ -d $COPY/derived/viz ]]; then
        NOTES[${#NOTES[@]} - 1]="${NOTES[${#NOTES[@]} - 1]}; figures: $(cd -- "$COPY/derived/viz" && ls -- *.png 2>/dev/null | tr '\n' ' ')"
    fi
    stage "7b view --print (real)" "$OUT/7b_view_real.log" "$PY" -m glassesdata_pipeline view "$COPY" --print || true
else
    skip "7 real recording" "no recording given (omen_smoke.sh REC)"
fi

# ------------------------------------------------------------------ summary

{
    printf 'GlassesData pipeline smoke test, %s\n' "$(date -Is)"
    printf 'log folder: %s\n\n' "$OUT"
    printf '%-42s %-5s %6s  %s\n' "stage" "" "s" "notes"
    for i in "${!NAMES[@]}"; do
        printf '%-42s %-5s %6s  %s\n' "${NAMES[$i]}" "${STATUSES[$i]}" "${TIMES[$i]}" "${NOTES[$i]}"
    done
    printf '\nrecordings: %s (gt backends), %s (Depth Anything V2), %s (SAM 2)%s\n' "$GT" "$DA" "$SAM" \
        "${REAL:+, $COPY (real)}"
    printf 'figures:    %s/derived/viz/*.png, %s/derived/eval.md, %s/derived/sim/replay.mp4\n' "$GT" "$GT" "$GT"
    printf 'rerun:      %s -m glassesdata_pipeline view %s\n' "$PY" "$GT"
} >"$OUT/summary.txt"
printf '\n%s== summary%s\n\n' "$BOLD" "$RESET"
cat -- "$OUT/summary.txt"
if [[ $FAILED -gt 0 ]]; then
    printf '\n%s%d stage(s) failed%s: send %s (all logs) for the report.\n' "$RED" "$FAILED" "$RESET" "$OUT"
    exit 1
fi
printf '\n%sall stages passed%s (WARN rows: see their logs).\n' "$GREEN" "$RESET"
exit 0
