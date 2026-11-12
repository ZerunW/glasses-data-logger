#!/usr/bin/env python3
"""Exercise the real broadcast server with synthetic local clients."""
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix="glasses-approval-") as temporary:
    directory = Path(temporary)
    main = directory / "main.swift"
    main.write_bytes((root / "SecurityTests/BroadcastApprovalTest.swift").read_bytes())
    executable = directory / "approval-test"
    sources = ["CameraAccess/Streaming/SecureTransport.swift",
               "CameraAccess/Streaming/CameraBroadcast.swift",
               "CameraAccess/Streaming/IucmMessage.swift",
               "CameraAccess/Streaming/IucmFrameParser.swift",
               "CameraAccess/Recording/MonotonicClock.swift"]
    subprocess.run(["xcrun", "swiftc", "-module-cache-path", str(directory / "cache"),
                    *[str(root / source) for source in sources], str(main), "-o", str(executable)], check=True)
    for mode in ("allow", "reject", "wrong"):
        subprocess.run([str(executable), mode], check=True, timeout=12)
