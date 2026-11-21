#!/usr/bin/env bash
# One-time setup of the GlassesData receiver on Ubuntu 24.04 (x86_64).
#
# Idempotent: run it again at any time. Every step checks first and only does what is
# missing. Each command is printed before it runs, and every step that needs sudo asks
# first (sudo then asks for your password itself).
#
#   Receiver/scripts/setup_ubuntu.sh             interactive (recommended the first time)
#   Receiver/scripts/setup_ubuntu.sh --yes       answer yes to every question
#
# Options:
#   -y, --yes      answer yes to every question
#   --no-vcam      skip the virtual camera (v4l2loopback and pyvirtualcam)
#   --no-tests     skip the final test suite (about two minutes)
#   --latest       install the newest versions requirements.txt allows, not the tested ones
#   -h, --help     show this help
#
# Environment: GLASSESDATA_VENV (default local/venv in the repository),
#              GLASSESDATA_PYTHON (default python3, Ubuntu 24.04's Python 3.12).
#
# Written on a Mac and checked with bash -n; see the README (Ubuntu 24.04) for what it does by hand.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECEIVER_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
readonly SCRIPT_DIR RECEIVER_DIR
REPO_DIR=$(cd -- "$RECEIVER_DIR/.." && pwd)
VENV=${GLASSESDATA_VENV:-$REPO_DIR/local/venv}
PYTHON=${GLASSESDATA_PYTHON:-python3}
VENV_PY="$VENV/bin/python"
readonly REPO_DIR VENV PYTHON VENV_PY
# The project keeps nothing outside its folder: pip's download cache too.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$REPO_DIR/local/cache/pip}

# The virtual camera device: /dev/video10, labelled GlassesData in Zoom, Meet and OBS.
readonly LOOPBACK_OPTIONS="devices=1 video_nr=10 card_label=GlassesData exclusive_caps=1"
readonly LOOPBACK_DEVICE=/dev/video10
readonly MODPROBE_CONF=/etc/modprobe.d/glassesdata-v4l2loopback.conf
readonly MODULES_LOAD_CONF=/etc/modules-load.d/glassesdata-v4l2loopback.conf
readonly MOK_KEY=/var/lib/shim-signed/mok/MOK.der
readonly PYVIRTUALCAM_TESTED=0.15.0

ME=$(id -un)
readonly ME
ASSUME_YES=0
NO_VCAM=0
NO_TESTS=0
LATEST=0
WARNINGS=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for argument in "$@"; do
    case $argument in
        -y | --yes) ASSUME_YES=1 ;;
        --no-vcam) NO_VCAM=1 ;;
        --no-tests) NO_TESTS=1 ;;
        --latest) LATEST=1 ;;
        -h | --help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n\n' "$argument" >&2; usage >&2; exit 2 ;;
    esac
done

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

# Print a command (on stderr, so it shows even when the command's output is redirected), then run it.
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

secure_boot_state() {
    local state value
    if command -v mokutil >/dev/null 2>&1 && state=$(mokutil --sb-state 2>/dev/null); then
        case $state in
            *enabled*) echo on; return ;;
            *disabled*) echo off; return ;;
        esac
    fi
    for value in /sys/firmware/efi/efivars/SecureBoot-*; do
        if [[ -r $value ]]; then
            # 4 bytes of attributes, then the value
            if [[ $(od -An -t u1 -j 4 -N 1 "$value" | tr -d ' ') == 1 ]]; then echo on; else echo off; fi
            return
        fi
    done
    echo unknown
}

# Write a root-owned file when its content differs. Args: path, content.
write_root_file() {
    local path=$1 content=$2
    if [[ -f $path && "$(cat -- "$path")" == "$content" ]]; then
        ok "$path is in place"
        return 0
    fi
    info "$path will contain:"
    printf '%s\n' "$content" | sed 's/^/        /'
    if ! ask "Write $path (sudo)?"; then
        info "not written"
        return 1
    fi
    # An explicit check: callers use "|| true", which turns set -e off inside this function.
    if printf '%s\n' "$content" | run_sudo tee -- "$path" >/dev/null; then
        ok "wrote $path"
    else
        warn "could not write $path"
        return 1
    fi
}

# ------------------------------------------------------------------ 1. this computer

check_system() {
    step "This computer"
    if [[ $EUID -eq 0 ]]; then
        die "run this as your own user, not with sudo: it asks for sudo itself where needed (the venv and the hand model belong to you)."
    fi
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
    if [[ $(uname -m) == x86_64 ]]; then
        ok "x86_64, kernel $(uname -r)"
    else
        warn "$(uname -m): only the x86_64 wheels of the Python packages were checked"
    fi
    info "Secure Boot: $(secure_boot_state)"
    info "receiver: $RECEIVER_DIR"
    info "venv:     $VENV"
    if command -v nvidia-smi >/dev/null 2>&1; then
        info "GPU: not needed. Decoding (PyAV) and hand tracking (MediaPipe) run on the CPU."
    fi
}

# ------------------------------------------------------------------ 2. Ubuntu packages

install_packages() {
    step "Ubuntu packages"
    # python3-venv/-dev: the venv (and headers should a wheel ever need building);
    # usbmuxd + libimobiledevice-utils: the USB link and idevicepair / idevice_id;
    # libgl1 + libglib2.0-0t64: what OpenCV's wheel loads (present on a desktop install).
    local wanted=(python3-venv python3-dev usbmuxd libimobiledevice-utils libgl1 libglib2.0-0t64)
    local missing=() package
    for package in "${wanted[@]}"; do
        is_installed "$package" || missing+=("$package")
    done
    if ! is_installed ffmpeg; then
        info "ffmpeg is optional: ffplay shows what the virtual camera sends, ffprobe inspects recordings."
        if ask "Also install ffmpeg?"; then
            missing+=(ffmpeg)
        fi
    fi
    if [[ ${#missing[@]} -eq 0 ]]; then
        ok "${wanted[*]} installed"
        return
    fi
    info "to install: ${missing[*]}"
    ask "Install them with apt (sudo)?" || die "the receiver needs these packages"
    run_sudo apt-get update
    if ! run_sudo apt-get install -y "${missing[@]}"; then
        die "apt could not install them. If it says \"Unable to locate package\", enable the universe component: sudo add-apt-repository universe"
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
    system_version=$(python_version "$PYTHON") || die "$PYTHON does not run"
    if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        die "$PYTHON is Python $system_version; the receiver needs 3.11 or newer (Ubuntu 24.04 has 3.12)"
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

    local constraints
    constraints=$(mktemp)
    # shellcheck disable=SC2064  # expand now: the path is fixed
    trap "rm -f -- '$constraints'" EXIT
    if [[ $LATEST -eq 0 ]]; then
        # The lower bounds in requirements*.txt are the versions the tests passed with; pin them.
        # Each has a wheel for CPython 3.12 on manylinux x86_64 (checked on PyPI).
        grep -hE '^[A-Za-z0-9_.-]+>=' "$RECEIVER_DIR/requirements.txt" "$RECEIVER_DIR/requirements-dev.txt" \
            | sed -E 's/>=/==/' >"$constraints"
        printf 'pyvirtualcam==%s\n' "$PYVIRTUALCAM_TESTED" >>"$constraints"
        info "installing the tested versions (--latest for the newest allowed):"
        sed 's/^/        /' "$constraints"
    fi
    local packages=(-r "$RECEIVER_DIR/requirements-dev.txt")
    if [[ $NO_VCAM -eq 0 ]]; then
        packages+=("pyvirtualcam>=0.15")
    fi
    run "$VENV_PY" -m pip install --disable-pip-version-check --prefer-binary -c "$constraints" "${packages[@]}"
    if "$VENV_PY" -m pip show --disable-pip-version-check opencv-python >/dev/null 2>&1; then
        warn "opencv-python is installed next to opencv-contrib-python: two copies of cv2. Remove it: $VENV_PY -m pip uninstall opencv-python"
    fi
    ok "Python packages installed"

    info "the hand model (about 8 MB, downloaded once into the cache folder):"
    if (cd -- "$RECEIVER_DIR" && run "$VENV_PY" -c 'from glassesdata_receiver.hands import ensure_model; print("   ", ensure_model())'); then
        ok "MediaPipe and OpenCV load, and the hand model is in place"
    else
        warn "could not load MediaPipe or fetch the hand model (see above); the receiver retries at start"
    fi
}

# ------------------------------------------------------------------ 4. the virtual camera

glassesdata_device_present() {
    local folder
    for folder in /sys/devices/virtual/video4linux/video*; do
        [[ -r $folder/name ]] || continue
        if [[ $(<"$folder/name") == GlassesData ]]; then
            return 0
        fi
    done
    return 1
}

print_mok_steps() {
    info "Secure Boot refuses this module: DKMS built it and signed it with this computer's own key (MOK),"
    info "which the firmware does not trust until you enrol it once:"
    info "  1. sudo mokutil --import $MOK_KEY        (choose a one-time password)"
    info "  2. Reboot. The blue \"Perform MOK management\" screen appears: Enroll MOK > Continue > Yes,"
    info "     type that password, Reboot. (Miss it and the key is not enrolled: repeat from 1.)"
    info "  3. Run this script again."
    info "  Check: mokutil --test-key $MOK_KEY   (says \"is already enrolled\" once done)"
}

load_loopback() {
    local output
    printf '   %s$ sudo modprobe v4l2loopback %s%s\n' "$DIM" "$LOOPBACK_OPTIONS" "$RESET" >&2
    # shellcheck disable=SC2086  # the options are separate words on purpose
    if output=$(sudo modprobe v4l2loopback $LOOPBACK_OPTIONS 2>&1); then
        ok "v4l2loopback loaded"
        return 0
    fi
    printf '%s\n' "$output" | sed 's/^/        /'
    if [[ $output == *"Key was rejected"* ]]; then
        print_mok_steps
    fi
    warn "v4l2loopback is not loaded; the receiver works without the virtual camera"
    return 1
}

setup_vcam() {
    step "Virtual camera (v4l2loopback)"
    if [[ $NO_VCAM -eq 1 ]]; then
        info "skipped (--no-vcam)"
        return
    fi
    info "Zoom, Meet, Teams and OBS can then use the glasses video as a webcam named \"GlassesData\"."
    if ! ask "Set up the virtual camera?"; then
        info "skipped; run this script again to add it later"
        return
    fi

    # The module. Ubuntu's kernel packages ship it, signed by Canonical, so Secure Boot accepts
    # it without any key enrolment; the newest kernels split it into its own package. DKMS is
    # the fallback for other kernels, and needs the MOK step under Secure Boot.
    local module kernel package
    kernel=$(uname -r)
    if ! module=$(modinfo -n v4l2loopback 2>/dev/null); then
        package="linux-main-modules-v4l2loopback-$kernel"
        if apt-cache show "$package" >/dev/null 2>&1; then
            info "this kernel's v4l2loopback module is in $package (signed by Ubuntu)"
            if ask "Install $package (sudo)?"; then
                run_sudo apt-get install -y "$package" || warn "apt could not install $package"
            fi
        else
            info "kernel $kernel has no packaged v4l2loopback module; v4l2loopback-dkms builds one."
            if [[ $(secure_boot_state) == on ]]; then
                info "Secure Boot is on: the installer creates a key (MOK) and may ask for a one-time password;"
                info "the module loads only after that key is enrolled at the next reboot (steps printed below if needed)."
            fi
            if ask "Install v4l2loopback-dkms (sudo)?"; then
                run_sudo apt-get install -y v4l2loopback-dkms || warn "apt could not install v4l2loopback-dkms"
            fi
        fi
        if ! module=$(modinfo -n v4l2loopback 2>/dev/null); then
            warn "no v4l2loopback module for kernel $kernel; the receiver works without the virtual camera"
            return
        fi
    fi
    local signer
    signer=$(modinfo -F signer v4l2loopback 2>/dev/null || true)
    ok "module $module (signed by: ${signer:-nobody})"
    if ! is_installed v4l2loopback-utils; then
        if ask "Install v4l2loopback-utils (v4l2loopback-ctl, optional) (sudo)?"; then
            run_sudo apt-get install -y v4l2loopback-utils || warn "apt could not install v4l2loopback-utils"
        fi
    fi

    # Load it at every start, with the GlassesData device.
    write_root_file "$MODPROBE_CONF" "# GlassesData receiver: its virtual camera, $LOOPBACK_DEVICE labelled GlassesData (Receiver/scripts/setup_ubuntu.sh)
options v4l2loopback $LOOPBACK_OPTIONS" || true
    write_root_file "$MODULES_LOAD_CONF" "# GlassesData receiver: load v4l2loopback at boot (options in $MODPROBE_CONF)
v4l2loopback" || true
    local other
    for other in /etc/modprobe.d/*.conf; do
        [[ -f $other && $other != "$MODPROBE_CONF" ]] || continue
        if grep -qE '^[[:space:]]*options[[:space:]]+v4l2loopback' "$other"; then
            warn "$other also sets v4l2loopback options; both apply and the last one read wins. Remove the one you do not want."
        fi
    done

    # Load it now.
    if [[ -d /sys/module/v4l2loopback ]]; then
        if glassesdata_device_present; then
            ok "v4l2loopback is loaded with the GlassesData device"
        else
            warn "v4l2loopback is loaded, but without a GlassesData device (loaded earlier with other options, by OBS for example)"
            if ask "Unload it and load it again with the GlassesData device (sudo; fails while an app uses one of its devices)?"; then
                if run_sudo modprobe -r v4l2loopback; then
                    load_loopback || true
                else
                    warn "it is in use: close OBS and video apps, or reboot, then run this script again"
                fi
            fi
        fi
    elif ask "Load v4l2loopback now (sudo)?"; then
        load_loopback || true
    fi

    if [[ -e $LOOPBACK_DEVICE ]]; then
        if [[ -w $LOOPBACK_DEVICE ]]; then
            ok "$LOOPBACK_DEVICE is writable by $ME"
        else
            warn "$LOOPBACK_DEVICE is not writable by $ME (a desktop login normally gets access; an SSH login does not)"
            if ask "Add $ME to the video group (sudo; takes effect at the next login)?"; then
                run_sudo usermod -aG video "$ME" || warn "usermod failed"
            fi
        fi
    fi
}

# ------------------------------------------------------------------ 5. the iPhone over USB

apple_device_plugged() {
    local folder
    for folder in /sys/bus/usb/devices/*; do
        [[ -r $folder/idVendor && -r $folder/idProduct ]] || continue
        if [[ $(<"$folder/idVendor") == 05ac && $(<"$folder/idProduct") == 12[9a][0-9a-f] ]]; then
            return 0
        fi
    done
    return 1
}

check_usb() {
    step "iPhone over USB"
    # udev starts usbmuxd when an iPhone is plugged in, and it asks the phone to show
    # "Trust This Computer" by itself; the pairing record ends up in /var/lib/lockdown.
    local udids udid attempt list=()
    udids=$(idevice_id -l 2>/dev/null || true)
    if [[ -z $udids ]]; then
        if apple_device_plugged; then
            warn "an iPhone is plugged in but usbmuxd does not list it"
            info "try: sudo systemctl restart usbmuxd; then see journalctl -u usbmuxd -n 30 (README > Ubuntu 24.04 > Troubleshooting)"
        else
            info "no iPhone connected. Later: plug it in, unlock it, tap Trust, then run Receiver/scripts/check_linux.sh"
        fi
        return
    fi
    # An array, not "while read ... <<<": the questions below must read from the terminal.
    mapfile -t list <<<"$udids"
    for udid in "${list[@]}"; do
        [[ -n $udid ]] || continue
        if idevicepair -u "$udid" validate >/dev/null 2>&1; then
            ok "iPhone $udid is paired with this computer"
            continue
        fi
        info "iPhone $udid is not paired yet. Unlock it: it shows \"Trust This Computer?\". Tap Trust and enter the passcode."
        for attempt in 1 2 3; do
            if ! ask "Pair now (idevicepair pair, attempt $attempt of 3)?"; then
                break
            fi
            if run idevicepair -u "$udid" pair; then
                ok "paired"
                break
            fi
            if [[ -t 0 && $ASSUME_YES -eq 0 ]]; then
                read -r -p "   Tap Trust on the iPhone (unlocked), then press Enter " _ || break
            else
                sleep 5
            fi
        done
        idevicepair -u "$udid" validate >/dev/null 2>&1 || warn "iPhone $udid is still not paired; run: idevicepair pair"
    done
}

# ------------------------------------------------------------------ 6. the firewall (Wi-Fi mode)

setup_firewall() {
    step "Firewall (only for Wi-Fi mode)"
    if ! command -v ufw >/dev/null 2>&1; then
        ok "ufw is not installed: nothing to open"
        return
    fi
    if ! grep -qs '^ENABLED=yes' /etc/ufw/ufw.conf; then
        ok "ufw is off (Ubuntu's default): nothing to open"
        return
    fi
    info "ufw is on. --wifi needs 7900/tcp (the link) and 5353/udp (Bonjour, so the iPhone finds this computer)."
    if ask "Allow them (sudo ufw allow; already present rules are skipped)?"; then
        run_sudo ufw allow 7900/tcp comment 'GlassesData link' || warn "ufw allow 7900/tcp failed"
        run_sudo ufw allow 5353/udp comment 'mDNS (Bonjour)' || warn "ufw allow 5353/udp failed"
    fi
}

# ------------------------------------------------------------------ 7. self-test

self_test() {
    step "Self-test"
    if (cd -- "$RECEIVER_DIR" && run "$VENV_PY" -m glassesdata_receiver --help >/dev/null); then
        ok "python -m glassesdata_receiver --help"
    else
        warn "the receiver does not start (see above)"
    fi
    if [[ $NO_TESTS -eq 1 ]]; then
        info "test suite skipped (--no-tests)"
        return
    fi
    info "the test suite (no device needed, loopback ports only; about two minutes):"
    if (cd -- "$RECEIVER_DIR" && run "$VENV_PY" -m pytest -q -p no:cacheprovider tests); then
        ok "all tests passed"
    else
        warn "some tests failed: keep the output above for the report"
    fi
}

check_system
install_packages
setup_venv
setup_vcam
check_usb
setup_firewall
self_test

step "Done"
if [[ $WARNINGS -gt 0 ]]; then
    info "$WARNINGS warning(s) above."
fi
info "Diagnostics at any time: $SCRIPT_DIR/check_linux.sh"
info "Start the receiver (iPhone on a cable, GlassesData open, Computer link (USB) on):"
info "  cd \"$RECEIVER_DIR\" && \"$VENV_PY\" -m glassesdata_receiver --usb --open"
