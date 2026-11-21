"""Visualisation of an episode.

* :mod:`.static` - matplotlib figures ``derived/viz/*.png`` (numpy + matplotlib; runs anywhere).
* :mod:`.rerun_export` - ``derived/viz/episode.rrd`` for the Rerun viewer (needs ``rerun-sdk``,
  pinned in ``requirements.txt``).
* :func:`main` - the ``view`` command: ``python -m glassesdata_pipeline view REC`` opens the
  recording's ``.rrd`` in the Rerun viewer, or prints how when it or ``rerun-sdk`` is missing. The
  viewer keeps its settings, UI state and analytics switch in the XDG config/data folders on
  Linux; :func:`viewer_environment` points them into ``local/cache/rerun-viewer`` (on macOS the
  viewer uses ``~/Library/Application Support/rerun``, which no variable moves).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from .. import paths

RERUN_PIN = "rerun-sdk==0.43.2"
RRD_NAME = "viz/episode.rrd"


def _rrd_for(target: Path) -> tuple[Path, Path | None]:
    """``(rrd path, recording folder or None)`` for a recording folder or an ``.rrd`` file."""
    if target.suffix == ".rrd":
        return target, None
    return target / "derived" / RRD_NAME, target


def viewer_environment() -> dict[str, str]:
    """The environment for the Rerun viewer: XDG config/data folders in ``local/cache/rerun-viewer``
    (variables that are already set win)."""
    env = dict(os.environ)
    home = paths.cache_dir() / "rerun-viewer"
    env.setdefault("XDG_CONFIG_HOME", str(home / "config"))
    env.setdefault("XDG_DATA_HOME", str(home / "data"))
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glassesdata_pipeline view",
        description="Open a recording's Rerun file (derived/viz/episode.rrd) in the Rerun viewer.")
    parser.add_argument("recording", help="recording folder (or an .rrd file)")
    parser.add_argument("--export", action="store_true", help="write derived/viz/episode.rrd first (needs rerun-sdk)")
    parser.add_argument("--static", action="store_true", help="(re)write the static PNG figures in derived/viz/ and list them")
    parser.add_argument("--print", dest="print_only", action="store_true", help="only print the command that opens the file")
    args = parser.parse_args(argv)

    target = Path(args.recording).expanduser()
    rrd, recording = _rrd_for(target)
    have_rerun = importlib.util.find_spec("rerun") is not None
    if args.static and recording is not None:
        from ..episode import Episode
        from .static import make_figures

        for name, path in make_figures(Episode(recording)).items():
            print(f"{name:<10} {path}")
    if args.export:
        if recording is None:
            print("--export needs a recording folder, not an .rrd file")
            return 2
        if not have_rerun:
            print(f"rerun-sdk is not installed here; install it with: pip install {RERUN_PIN}")
            return 2
        from ..episode import Episode
        from .rerun_export import export_rrd

        ep = Episode(recording)
        info = export_rrd(ep, ep.derived / RRD_NAME, ep.config.section("viz"))
        print(f"wrote {info['path']}")
    if not rrd.is_file():
        print(f"No Rerun recording at {rrd}.")
        if recording is not None:
            print(f"Create it on a machine with rerun-sdk ({RERUN_PIN}):\n"
                  f"  python -m glassesdata_pipeline run {recording} --steps viz\n"
                  f"  or: python -m glassesdata_pipeline view {recording} --export")
            pngs = sorted((recording / "derived" / "viz").glob("*.png"))
            if pngs:
                print("Static figures:")
                for png in pngs:
                    print(f"  {png}")
        return 2
    command = [sys.executable, "-m", "rerun", str(rrd)]
    if args.print_only:
        print(" ".join(command))
        return 0
    if not have_rerun:
        print(f"rerun-sdk is not installed here. Install it (pip install {RERUN_PIN}) and run:\n  rerun {rrd}")
        return 2
    return subprocess.call(command, env=viewer_environment())
