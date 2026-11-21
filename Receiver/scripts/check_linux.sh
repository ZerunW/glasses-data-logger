#!/usr/bin/env bash
# Diagnostics for the GlassesData receiver on Linux (written for Ubuntu 24.04).
#
# Read-only: it changes nothing and needs no sudo. Run it when something does not work and
# keep its output for a report. Every problem line says what to do.
#
#   Receiver/scripts/check_linux.sh
#
# Exit status: 0 when no check failed (warnings allowed), 1 otherwise.
# Environment: GLASSESDATA_VENV (default local/venv in the repository).
#
# Written on a Mac and checked with bash -n; not yet run on Ubuntu.

# Not -e: a diagnostic must go on to the next check when one of them fails.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECEIVER_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd -- "$RECEIVER_DIR/.." && pwd)
VENV=${GLASSESDATA_VENV:-$REPO_DIR/local/venv}
VENV_PY="$VENV/bin/python"
readonly SCRIPT_DIR RECEIVER_DIR REPO_DIR VENV VENV_PY
readonly LINK_PORT=7900 WEB_PORT=8765
readonly MODPROBE_CONF=/etc/modprobe.d/glassesdata-v4l2loopback.conf
readonly MODULES_LOAD_CONF=/etc/modules-load.d/glassesdata-v4l2loopback.conf

case ${1:-} in
    -h | --help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) printf 'Unknown option: %s (try --help)\n' "$1" >&2; exit 2 ;;
esac

if [[ -t 1 ]]; then
    BOLD=$'\033[1m' GREEN=$'\033[32m' YELLOW=$'\033[33m' RED=$'\033[31m' RESET=$'\033[0m'
else
    BOLD="" GREEN="" YELLOW="" RED="" RESET=""
fi
readonly BOLD GREEN YELLOW RED RESET

FAILS=0
WARNS=0
section() { printf '\n%s== %s%s\n' "$BOLD" "$*" "$RESET"; }
info() { printf '   %-6s%s\n' "" "$*"; }
ok() { printf '   %sok%s    %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '   %swarn%s  %s\n' "$YELLOW" "$RESET" "$*"; WARNS=$((WARNS + 1)); }
fail() { printf '   %sFAIL%s  %s\n' "$RED" "$RESET" "$*"; FAILS=$((FAILS + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------------ this computer

section "This computer"
if [[ -r /etc/os-release ]]; then
    # shellcheck source=/dev/null
    info "$(. /etc/os-release && printf '%s' "${PRETTY_NAME:-unknown}"), kernel $(uname -r), $(uname -m)"
else
    info "kernel $(uname -r), $(uname -m)"
fi
if [[ $EUID -eq 0 ]]; then
    warn "running as root: run the receiver and this check as your own user (no sudo)"
fi
if [[ -n ${WAYLAND_DISPLAY:-} || -n ${DISPLAY:-} ]]; then
    ok "graphical session (${XDG_SESSION_TYPE:-unknown}): --open can start the browser"
else
    info "no graphical session here (SSH?): --open does nothing; open the printed WEB_URL in a browser"
fi
secure_boot="unknown"
if have mokutil && state=$(mokutil --sb-state 2>/dev/null); then
    case $state in
        *enabled*) secure_boot=on ;;
        *disabled*) secure_boot=off ;;
    esac
fi
info "Secure Boot: $secure_boot"
if have nvidia-smi; then
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    info "GPU: ${gpu:-NVIDIA} (not needed: decoding and hand tracking run on the CPU)"
else
    info "GPU: not needed (decoding and hand tracking run on the CPU)"
fi

# ------------------------------------------------------------------ Python

section "Python environment ($VENV)"
if [[ ! -x $VENV_PY ]]; then
    fail "no venv at $VENV: run $SCRIPT_DIR/setup_ubuntu.sh"
else
    ok "$("$VENV_PY" -c 'import sys; print("Python %d.%d.%d" % sys.version_info[:3])' 2>&1)"
    # One line per module: "ok|warn|fail name detail". pyvirtualcam is optional.
    report=$(cd -- "$RECEIVER_DIR" && "$VENV_PY" - <<'PY' 2>&1
import importlib
import importlib.metadata
optional = {"pyvirtualcam"}
distributions = {"cv2": "opencv-contrib-python"}
for name in ("numpy", "cv2", "av", "mediapipe", "cryptography", "zeroconf", "ifaddr", "qrcode", "pyvirtualcam"):
    try:
        module = importlib.import_module(name)
    except Exception as error:  # ImportError, or OSError for a missing system library
        print("warn" if name in optional else "fail", name, f"{type(error).__name__}: {error}")
        continue
    try:
        version = importlib.metadata.version(distributions.get(name, name))
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "")
    print("ok", name, version)
try:
    from glassesdata_receiver import paths
    print("paths", paths.recordings_dir(), paths.state_dir(), paths.cache_dir(), sep="\t")
except Exception as error:
    print("fail", "glassesdata_receiver", f"{type(error).__name__}: {error}")
PY
)
    recordings="" state_dir="" cache_dir=""
    while IFS= read -r line; do
        case $line in
            "ok "*) ok "${line#ok }" ;;
            "warn pyvirtualcam"*) warn "pyvirtualcam missing (only the virtual camera needs it): $VENV_PY -m pip install pyvirtualcam" ;;
            "warn "*) warn "${line#warn }" ;;
            "fail "*)
                fail "${line#fail }"
                if [[ $line == *libGL* || $line == *libgthread* ]]; then
                    info "OpenCV needs two system libraries: sudo apt install libgl1 libglib2.0-0t64"
                fi
                ;;
            paths$'\t'*) IFS=$'\t' read -r _ recordings state_dir cache_dir <<<"$line" ;;
            *) info "$line" ;;
        esac
    done <<<"$report"
    if "$VENV_PY" -m pip show --disable-pip-version-check opencv-python >/dev/null 2>&1; then
        warn "opencv-python is installed next to opencv-contrib-python (two copies of cv2): $VENV_PY -m pip uninstall opencv-python"
    fi

    section "Folders"
    if [[ -n $recordings ]]; then
        if [[ -d $recordings ]]; then
            if [[ -w $recordings ]]; then
                ok "recordings: $recordings ($(df -h --output=avail -- "$recordings" 2>/dev/null | tail -n 1 | tr -d ' ') free)"
            else
                fail "recordings: $recordings is not writable by $(id -un)"
            fi
        else
            info "recordings: $recordings (created at the first recording)"
        fi
        info "state (Wi-Fi identity, calibration, body measurements): $state_dir"
        if [[ -s $cache_dir/hand_landmarker.task ]]; then
            ok "hand model: $cache_dir/hand_landmarker.task"
        else
            info "hand model: not downloaded yet (the receiver fetches it into $cache_dir at its first start)"
        fi
    fi
fi

# ------------------------------------------------------------------ USB: usbmuxd and the iPhone

section "iPhone over USB (usbmuxd)"
apple_devices=()
for folder in /sys/bus/usb/devices/*; do
    [[ -r $folder/idVendor && -r $folder/idProduct ]] || continue
    if [[ $(<"$folder/idVendor") == 05ac && $(<"$folder/idProduct") == 12[9a][0-9a-f] ]]; then
        product="Apple device"
        [[ -r $folder/product ]] && product=$(<"$folder/product")
        speed=""
        [[ -r $folder/speed ]] && speed=", $(<"$folder/speed") Mbit/s"
        apple_devices+=("$product (05ac:$(<"$folder/idProduct")$speed)")
    fi
done
if have usbmuxd || [[ -x /usr/sbin/usbmuxd || -x /sbin/usbmuxd ]]; then
    ok "usbmuxd is installed"
else
    fail "usbmuxd is not installed: sudo apt install usbmuxd libimobiledevice-utils"
fi
service="unknown"
if have systemctl; then
    service=$(systemctl is-active usbmuxd 2>/dev/null)
    service=${service:-unknown}
fi
if [[ ${#apple_devices[@]} -eq 0 ]]; then
    info "no iPhone on the USB bus. usbmuxd ($service) runs only while one is plugged in: plug it in with a cable and unlock it"
else
    ok "on the USB bus: ${apple_devices[*]}"
    if [[ $service == active ]]; then
        ok "usbmuxd is running"
    else
        fail "usbmuxd is $service although an iPhone is plugged in: sudo systemctl restart usbmuxd; journalctl -u usbmuxd -n 30"
    fi
fi
socket=/var/run/usbmuxd
if [[ -S $socket ]]; then
    mode=$(stat -L -c '%a %U:%G' -- "$socket" 2>/dev/null)
    if [[ -r $socket && -w $socket ]]; then
        ok "socket $socket ($mode): this user can use it"
    else
        fail "socket $socket ($mode) is not usable by $(id -un) (usbmuxd normally makes it 0666)"
    fi
elif [[ ${#apple_devices[@]} -gt 0 ]]; then
    fail "no socket at $socket although an iPhone is plugged in"
fi
if have idevice_id; then
    udids=$(idevice_id -l 2>/dev/null)
    if [[ -n $udids ]]; then
        mapfile -t list <<<"$udids"
        for udid in "${list[@]}"; do
            [[ -n $udid ]] || continue
            if result=$(idevicepair -u "$udid" validate 2>&1); then
                ok "iPhone $udid is listed and paired (trusted)"
            else
                warn "iPhone $udid is listed but not paired: unlock it, tap Trust, or run: idevicepair pair ($result)"
            fi
        done
    elif [[ ${#apple_devices[@]} -gt 0 ]]; then
        fail "the iPhone is plugged in but usbmuxd lists no device: unlock it, try another cable or port, sudo systemctl restart usbmuxd"
    fi
else
    warn "idevice_id is missing (pairing checks): sudo apt install libimobiledevice-utils"
fi
if have journalctl; then
    crashes=$(journalctl -u usbmuxd --since "7 days ago" --no-pager -q 2>/dev/null | grep -ciE 'segfault|SEGV|core-dump|dumped core')
    if [[ ${crashes:-0} -gt 0 ]]; then
        warn "usbmuxd crashed $crashes time(s) in the last 7 days. Known on Ubuntu 24.04 with usbmuxd 1.1.1 and USB-C to USB-C cables"
        info "(libimobiledevice/usbmuxd issue #246); a USB-A to USB-C cable was reported to work. See journalctl -u usbmuxd"
    fi
fi

# ------------------------------------------------------------------ the virtual camera

section "Virtual camera (v4l2loopback)"
if module=$(modinfo -n v4l2loopback 2>/dev/null); then
    version=$(modinfo -F version v4l2loopback 2>/dev/null)
    signer=$(modinfo -F signer v4l2loopback 2>/dev/null)
    ok "module ${version:+$version }at $module (signed by: ${signer:-nobody})"
    if [[ $secure_boot == on && -z $signer ]]; then
        warn "Secure Boot is on and the module is unsigned: modprobe will say \"Key was rejected by service\""
    elif [[ $secure_boot == on && $module == */updates/dkms/* && -r /var/lib/shim-signed/mok/MOK.der ]] && have mokutil; then
        # A DKMS build is signed with this computer's own key (MOK); the firmware must know it.
        enrolled=$(mokutil --test-key /var/lib/shim-signed/mok/MOK.der 2>&1)
        case $enrolled in
            *"is already enrolled"*) ok "the DKMS signing key (MOK) is enrolled" ;;
            *"not enrolled"*) warn "the DKMS signing key is not enrolled: sudo mokutil --import /var/lib/shim-signed/mok/MOK.der, reboot, Enroll MOK (README)" ;;
            *) info "MOK check: $enrolled" ;;
        esac
    fi
else
    warn "no v4l2loopback module for kernel $(uname -r): run $SCRIPT_DIR/setup_ubuntu.sh (README > Ubuntu 24.04 > Virtual camera)"
fi
loopback_devices=()
own_device=""
for folder in /sys/devices/virtual/video4linux/video*; do
    [[ -d $folder ]] || continue
    [[ -e $folder/max_openers || -e $folder/buffers || -e $folder/format ]] || continue
    label=""
    [[ -r $folder/name ]] && label=$(<"$folder/name")
    device="/dev/${folder##*/}"
    loopback_devices+=("$device ($label)")
    if [[ $label == GlassesData && -z $own_device ]]; then
        own_device=$device
    fi
done
if [[ -d /sys/module/v4l2loopback ]]; then
    if [[ ${#loopback_devices[@]} -eq 0 ]]; then
        warn "v4l2loopback is loaded without a device: sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=GlassesData exclusive_caps=1"
    else
        ok "loaded; devices: ${loopback_devices[*]}"
    fi
    if [[ -r /sys/module/v4l2loopback/parameters/exclusive_caps ]]; then
        info "exclusive_caps per device: $(</sys/module/v4l2loopback/parameters/exclusive_caps) (Y is needed for Chrome and Zoom to list it)"
    fi
else
    warn "v4l2loopback is not loaded: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=GlassesData exclusive_caps=1"
fi
if [[ -n $own_device ]]; then
    if [[ -w $own_device ]]; then
        ok "$own_device (GlassesData) is writable by $(id -un)"
    else
        warn "$own_device is not writable by $(id -un): sudo usermod -aG video $(id -un), then log out and in"
    fi
    if have fuser; then
        # fuser prints the PIDs on stdout, with access letters such as "m" (mmap) glued to them
        users=$(fuser -- "$own_device" 2>/dev/null | grep -oE '[0-9]+' | paste -sd, -)
        if [[ -n $users ]]; then
            info "$own_device is open in: $(ps -o comm= -p "$users" 2>/dev/null | paste -sd, -) (only one program can write to it)"
        fi
    fi
elif [[ ${#loopback_devices[@]} -gt 0 ]]; then
    info "no device labelled GlassesData: the receiver uses the first v4l2loopback device, or pass --virtual-camera-device"
fi
for file in "$MODPROBE_CONF" "$MODULES_LOAD_CONF"; do
    if [[ -f $file ]]; then
        ok "$file is in place (loaded at every start)"
    else
        info "$file is missing: v4l2loopback is not loaded at boot (setup_ubuntu.sh writes it)"
    fi
done

# ------------------------------------------------------------------ network: ports and firewall

section "Network"
for port in "$WEB_PORT" "$LINK_PORT"; do
    what="the web page"
    [[ $port == "$LINK_PORT" ]] && what="the Wi-Fi link (--wifi)"
    if have ss; then
        listeners=$(ss -Hltnp "sport = :$port" 2>/dev/null)
        if [[ -z $listeners ]]; then
            ok "port $port is free ($what)"
        else
            owner=$(grep -o 'users:(("[^"]*"' <<<"$listeners" | head -n 1 | sed 's/users:(("//; s/"$//')
            warn "port $port ($what) is in use${owner:+ by $owner} (a receiver already running?). Stop it, or pick another port (--web-port / --port)"
        fi
    fi
done
if have ip; then
    addresses=$(ip -4 -brief address show scope global 2>/dev/null | awk '{print $1 " " $3}' | paste -sd ';' -)
    info "IPv4: ${addresses:-none} (the Wi-Fi pairing code carries these; campus networks such as eduroam often block phone-to-laptop traffic: use USB or a hotspot)"
fi
if have systemctl && [[ $(systemctl is-active avahi-daemon 2>/dev/null) == active ]]; then
    info "avahi-daemon is running: fine, the receiver's own Bonjour responder shares port 5353 with it"
fi
if have ufw; then
    if grep -qs '^ENABLED=yes' /etc/ufw/ufw.conf; then
        warn "ufw is on: --wifi needs sudo ufw allow $LINK_PORT/tcp and sudo ufw allow 5353/udp (check: sudo ufw status). USB mode needs nothing"
    else
        ok "ufw is off (Ubuntu's default): nothing blocks --wifi"
    fi
else
    info "ufw is not installed"
fi

# ------------------------------------------------------------------ summary

section "Summary"
if [[ $FAILS -gt 0 ]]; then
    printf '   %s%d problem(s)%s and %d warning(s) above.\n' "$RED" "$FAILS" "$RESET" "$WARNS"
    exit 1
fi
printf '   %sno problems%s, %d warning(s).\n' "$GREEN" "$RESET" "$WARNS"
exit 0
