#!/usr/bin/env python3
"""Pairing protocol checks using ephemeral synthetic local peers."""
import subprocess
import tempfile
from pathlib import Path
root = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix="glasses-pair-test-") as tmp:
    directory = Path(tmp)
    main = directory / "main.swift"
    main.write_bytes((root / "SecurityTests/PairingProtocolTest.swift").read_bytes())
    binary = directory / "test"
    sources = ["CameraAccess/Streaming/PairingCryptography.swift", "CameraAccess/Streaming/PairingConnection.swift"]
    subprocess.run(["xcrun", "swiftc", "-module-cache-path", str(directory / "cache"),
                    *[str(root / path) for path in sources], str(main), "-o", str(binary)], check=True)
    for scenario in ("allow", "wrong", "tamper", "pending"):
        subprocess.run([str(binary), scenario], timeout=12, check=True)

    host_main = directory / "main.swift"
    host_main.write_bytes((root / "SecurityTests/PairingHostTest.swift").read_bytes())
    host_binary = directory / "host-test"
    subprocess.run(["xcrun", "swiftc", "-module-cache-path", str(directory / "cache"),
                    *[str(root / path) for path in sources],
                    str(root / "CameraAccess/Streaming/PairingHost.swift"),
                    str(host_main), "-o", str(host_binary)], check=True)
    subprocess.run([str(host_binary)], timeout=12, check=True)
