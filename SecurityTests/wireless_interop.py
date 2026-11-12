#!/usr/bin/env python3
"""Run the GlassesData TLS relay against a local TetherCam source checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('tethercam_source', type=Path, help='Path to a TetherCam source checkout')
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix='glasses-wireless-interop-') as tmp:
    work = Path(tmp)
    tethercam = work / 'tethercam'
    shutil.copytree(args.tethercam_source, tethercam,
                    ignore=shutil.ignore_patterns('.git', '.build', '.DS_Store'))
    relay_source = (root / 'MacRelay/main.swift').read_text()
    relay = relay_source.split('final class Relay: @unchecked Sendable {', 1)[1].split('\nstruct ContentView:', 1)[0]
    (work / 'Relay.swift').write_text('import Foundation\nimport Network\n\nfinal class Relay: @unchecked Sendable {' + relay)
    shutil.copy(root / 'SecurityTests/WirelessInteropSender.swift', work / 'main.swift')
    sources = ['CameraAccess/Streaming/SecureTransport.swift', 'CameraAccess/Streaming/CameraBroadcast.swift',
               'CameraAccess/Streaming/IucmMessage.swift', 'CameraAccess/Streaming/IucmFrameParser.swift',
               'CameraAccess/Recording/MonotonicClock.swift', 'CameraAccess/Media/VideoCaptureHandler.swift']
    sender = work / 'sender'
    subprocess.run(['xcrun', 'swiftc', '-suppress-warnings', '-module-cache-path', str(work / 'cache'),
                    *[str(root / s) for s in sources], str(work / 'Relay.swift'), str(work / 'main.swift'),
                    '-o', str(sender)], check=True)
    tests = tethercam / 'mac-app/Core/Tests/TetherCamCoreTests'
    shutil.copy(root / 'SecurityTests/WirelessInteropTest.swift', tests / 'GlassesSenderInteropTests.swift')
    env = os.environ.copy()
    env['GLASSESDATA_WIRELESS_SENDER'] = str(sender)
    env['GLASSESDATA_SAMPLE_VIDEO'] = str(root / 'CameraAccess/TestResources/plant.mp4')
    subprocess.run(['swift', 'test', '--package-path', str(tethercam / 'mac-app/Core'),
                    '--filter', 'GlassesSenderInteropTests'], env=env, check=True)
