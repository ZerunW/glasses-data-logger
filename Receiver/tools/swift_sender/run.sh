#!/bin/bash
# Builds a macOS stand-in for the iPhone app from the app's real link sources and runs it.
# The binary goes to a temporary directory in the repository's gitignored local/tmp.
#
#   Receiver/tools/swift_sender/run.sh --usb [--port P] [--seconds N] [VIDEO]
#   Receiver/tools/swift_sender/run.sh --wifi 'gdpair://v1?...' [--host H] [--seconds N] [VIDEO]
#
# VIDEO defaults to CameraAccess/TestResources/plant.mp4. Set SWIFT_SENDER_BUILD_DIR to
# reuse a build directory between runs.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
app="$repo/CameraAccess"

build_dir="${SWIFT_SENDER_BUILD_DIR:-}"
if [[ -z "$build_dir" ]]; then
  mkdir -p "$repo/local/tmp"
  build_dir="$(mktemp -d "$repo/local/tmp/swift_sender.XXXXXX")"
  trap 'rm -rf "$build_dir"' EXIT
fi
mkdir -p "$build_dir"
binary="$build_dir/swift_sender"

sources=(
  "$app/Streaming/LinkProtocol.swift"
  "$app/Streaming/LinkPairing.swift"
  "$app/Streaming/LinkTLS.swift"
  "$app/Streaming/ComputerLink.swift"
  "$app/Recording/MonotonicClock.swift"
  "$here/main.swift"
)

newest_source="$(ls -t "${sources[@]}" | head -1)"
if [[ ! -x "$binary" || "$newest_source" -nt "$binary" ]]; then
  echo "[run.sh] building $binary" >&2
  xcrun swiftc -swift-version 6 -O -target "$(uname -m)-apple-macos14.0" \
    -module-name swift_sender -o "$binary" "${sources[@]}"
fi

args=("$@")
has_video=0
skip_next=0
for argument in "${args[@]+"${args[@]}"}"; do
  if (( skip_next )); then skip_next=0; continue; fi
  case "$argument" in
    --wifi|--port|--host|--seconds) skip_next=1 ;;
    --*) ;;
    *) has_video=1 ;;
  esac
done
if (( ! has_video )); then
  args+=("$app/TestResources/plant.mp4")
fi

"$binary" "${args[@]}"
