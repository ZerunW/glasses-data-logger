#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
APP="$ROOT/MacRelay/build/GlassesData Mac.app"
TETHERCAM_APP="${TETHERCAM_APP:-/Applications/TetherCam.app}"
BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/glasses-mac-build.XXXXXXXX")
trap 'rm -rf "$BUILD_DIR"' EXIT HUP INT TERM
codesign --verify --deep --strict -R '=anchor apple generic and identifier "at.gotzendorfer.tethercam.mac" and certificate leaf[subject.OU] = "G3QZ66475M"' "$TETHERCAM_APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$APP/Contents/Helpers"
rm -rf "$APP/Contents/Helpers/TetherCam.app"
ditto "$TETHERCAM_APP" "$APP/Contents/Helpers/TetherCam.app"
cp "$ROOT/CameraAccess/Streaming/TetherCam-LICENSE.txt" "$APP/Contents/Resources/TetherCam-LICENSE.txt"
xcrun swiftc -parse-as-library -module-cache-path "$BUILD_DIR/cache" "$ROOT/MacRelay/Tools/DrawIcons.swift" -o "$BUILD_DIR/draw-icons"
"$BUILD_DIR/draw-icons" "$BUILD_DIR/icons" "$ROOT/MacRelay/Assets/GlassesData-Mark-Reference.png"
cp "$BUILD_DIR/icons/GlassesData-Mac.png" "$ROOT/MacRelay/Assets/GlassesData-Mac.png"
cp "$BUILD_DIR/icons/GlassesData-iPhone.png" "$ROOT/CameraAccess/Assets.xcassets/AppIcon.appiconset/GlassesData-iPhone.png"
sips -z 512 512 "$BUILD_DIR/icons/GlassesData-Mac.png" --out "$BUILD_DIR/icons/GlassesData-Mac-512.png" >/dev/null
python3 - "$BUILD_DIR/icons/GlassesData-Mac.png" "$BUILD_DIR/icons/GlassesData-Mac-512.png" "$APP/Contents/Resources/AppIcon.icns" <<'PY'
import struct
import sys

def chunk(kind, path):
    data = open(path, "rb").read()
    return kind + struct.pack(">I", len(data) + 8) + data

payload = chunk(b"ic09", sys.argv[2]) + chunk(b"ic10", sys.argv[1])
with open(sys.argv[3], "wb") as output:
    output.write(b"icns" + struct.pack(">I", len(payload) + 8) + payload)
PY
xcrun swiftc -parse-as-library -module-cache-path "$BUILD_DIR/cache" "$ROOT/CameraAccess/Streaming/SecureTransport.swift" "$ROOT/CameraAccess/Streaming/PairingCryptography.swift" "$ROOT/CameraAccess/Streaming/PairingConnection.swift" "$ROOT/CameraAccess/Streaming/PairingTrustStore.swift" "$ROOT/MacRelay/main.swift" -o "$APP/Contents/MacOS/GlassesDataMac"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>GlassesDataMac</string>
<key>CFBundleIdentifier</key><string>com.glassesdata.mac</string>
<key>CFBundleName</key><string>GlassesData Mac</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleIconFile</key><string>AppIcon</string>
<key>NSHighResolutionCapable</key><true/>
<key>NSBonjourServices</key><array><string>_glassesdata._tcp</string><string>_gdpair._tcp</string></array>
<key>NSLocalNetworkUsageDescription</key><string>Connect to your iPhone to receive encrypted camera video.</string>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP"
printf 'Built %s\n' "$APP"
