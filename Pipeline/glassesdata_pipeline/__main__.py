"""Command line: ``python -m glassesdata_pipeline <command> ...``

Commands:

* ``run REC [--steps a,b] [--from x] [--until y] [--force] [--no-deps] [--dry-run]``
* ``synth OUT [--seconds 8] [--seed 0] [--size 360x640] [--draw-hands] [--overwrite]``
* ``eval REC [--no-plots]``
* ``list-steps [REC]``
* ``view``, ``train``, ``eval-charuco``: delegate to ``glassesdata_pipeline.viz``,
  ``glassesdata_pipeline.rl.train`` and ``glassesdata_pipeline.eval.charuco`` (each must expose
  ``main(argv: list[str]) -> int``); they print "not implemented yet" until those modules exist.

``--config FILE`` and ``--set section.key=value`` (repeatable) work with every command that
reads the configuration (run, synth, eval, list-steps).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from typing import Any

from . import __version__
from .config import load_config, overrides_from_cli
from .errors import PipelineError

log = logging.getLogger("glassesdata_pipeline")

LAZY_COMMANDS = {
    "view": ("glassesdata_pipeline.viz", "interactive / Rerun visualisation of a recording"),
    "train": ("glassesdata_pipeline.rl.train", "RL training on a reconstructed episode"),
    "eval-charuco": ("glassesdata_pipeline.eval.charuco", "evaluation against the on-screen ChArUco board"),
}


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", metavar="FILE", help="extra TOML file merged over configs/ and pipeline.toml")
    parser.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override one value (parsed as TOML, else a string); repeatable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m glassesdata_pipeline",
                                     description="Offline pipeline for Ray-Ban Meta recordings (Docs/PIPELINE.md).")
    parser.add_argument("--version", action="version", version=f"glassesdata_pipeline {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run processing steps on a recording folder")
    p.add_argument("recording")
    p.add_argument("--steps", help="comma-separated steps to run (in pipeline order)")
    p.add_argument("--from", dest="from_", metavar="STEP", help="start at this step")
    p.add_argument("--until", metavar="STEP", help="stop after this step")
    p.add_argument("--force", action="store_true", help="re-run the selected steps even when cached")
    p.add_argument("--no-deps", action="store_true", help="do not add required steps that are not selected")
    p.add_argument("--dry-run", action="store_true", help="only show what would run")
    _add_config_args(p)

    p = sub.add_parser("synth", help="write a synthetic recording with ground truth")
    p.add_argument("out", help="output recording folder (created; must be empty unless --overwrite)")
    p.add_argument("--seconds", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--size", help="WIDTHxHEIGHT, e.g. 360x640 or 720x1280")
    p.add_argument("--draw-hands", action="store_true", default=None, help="render hands and arms as capsules")
    p.add_argument("--overwrite", action="store_true", help="replace an existing folder")
    _add_config_args(p)

    p = sub.add_parser("eval", help="compare derived/ with gt/ (synthetic recordings)")
    p.add_argument("recording")
    p.add_argument("--no-plots", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("list-steps", help="list the steps and whether they are implemented")
    p.add_argument("recording", nargs="?", help="also show the cache state for this recording")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    _add_config_args(p)

    for name, (module, text) in LAZY_COMMANDS.items():
        p = sub.add_parser(name, help=f"{text} (module {module})", add_help=False)
        p.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _setup_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def cmd_run(args: argparse.Namespace) -> int:
    from .episode import Episode
    from .runner import Runner

    ep = Episode(args.recording, overrides_from_cli(args.config, args.sets))
    steps = [s.strip() for s in args.steps.split(",") if s.strip()] if args.steps else None
    started = time.perf_counter()
    report = Runner(ep).run(steps=steps, from_=args.from_, until=args.until, force=args.force,
                            deps=not args.no_deps, dry_run=args.dry_run)
    print(report.table())
    print(f"total {time.perf_counter() - started:.2f} s")
    return 0 if report.ok else 1


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import generate

    cfg = load_config(config_file=args.config, sets=args.sets)
    summary = generate(args.out, seconds=args.seconds, seed=args.seed, size=args.size,
                       draw_hands=args.draw_hands, config=cfg, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .episode import Episode
    from .eval.report import run_eval

    ep = Episode(args.recording, overrides_from_cli(args.config, args.sets))
    result = run_eval(ep, plots=False if args.no_plots else None)
    for group, value in result["metrics"].items():
        status = value.get("status", "ok")
        print(f"{group:<12} {status:<8} {value.get('reason', '')}")
    print(f"wrote {result['json']} and {result['markdown']}")
    for png in result["plots"]:
        print(f"      {png}")
    return 0


def cmd_list_steps(args: argparse.Namespace) -> int:
    from .runner import Runner, describe_steps

    if args.recording:
        from .episode import Episode

        rows = Runner(Episode(args.recording, overrides_from_cli(args.config, args.sets))).describe()
    else:
        rows = describe_steps()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'step':<10} {'status':<20} {'version':<8} {'requires':<24} {'cache' if args.recording else ''}")
    for r in rows:
        requires = ",".join(r["requires"]) or "-"
        extra = r.get("cache", "") if args.recording else ""
        version = "" if r["version"] is None else str(r["version"])
        print(f"{r['name']:<10} {r['status']:<20} {version:<8} {requires:<24} {extra}")
        if r.get("error"):
            print(f"{'':<10} error: {r['error']}")
    return 0


def cmd_lazy(name: str, argv: list[str]) -> int:
    module_name, text = LAZY_COMMANDS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and module_name.startswith(error.name):
            print(f"'{name}' is not implemented yet ({text}; expected module {module_name} with main(argv)).")
            return 2
        raise
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        print(f"'{name}': {module_name} has no main(argv) yet.")
        return 2
    return int(main_fn(argv) or 0)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args)
    try:
        if args.command in LAZY_COMMANDS:
            return cmd_lazy(args.command, args.args)
        handler: dict[str, Any] = {
            "run": cmd_run, "synth": cmd_synth, "eval": cmd_eval, "list-steps": cmd_list_steps,
        }
        return handler[args.command](args)
    except PipelineError as error:
        log.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
