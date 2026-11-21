"""Step registry, dependency order, caching and logging (PIPELINE.md sections 5 and 7).

Steps live in ``glassesdata_pipeline/steps/<name>.py`` and are listed in the fixed :data:`ORDER`.
A missing module is reported as "not implemented yet" (never a crash); a module that fails to
import (e.g. a heavy dependency imported at module level) is "unavailable" with the error.

Step module contract (see ``steps/__init__.py``)::

    NAME = "hand3d"                      # must equal the module name
    VERSION = 1                          # bump to invalidate cached outputs
    REQUIRES = ["hands2d", "depth"]      # step names
    INPUTS = ["derived/hands2d.npz", ...]  # relative to the RECORDING folder (or a callable cfg -> list)
    OUTPUTS = ["hand3d.npz"]             # relative to derived/
    CONFIG_SECTIONS = ["hand3d"]         # config sections that affect the result (default [NAME])
    def run(ep, cfg) -> dict: ...        # cfg = the full merged config; returns a short summary

Caching: a step is skipped when all OUTPUTS exist and the meta stored in every meta-carrying
output (``.npz`` ``meta_json`` / ``.json`` ``"meta"``) has the step's VERSION, the same params
(the CONFIG_SECTIONS of the merged config) and the same INPUTS ``[size, mtime_ns]``. Steps without
a meta-carrying output fall back to the runner's record ``derived/_runs/<step>.json`` (written
after every successful run). ``force`` re-runs the explicitly selected steps.

Selection: ``steps=[...]``, ``from_``, ``until`` choose steps in ORDER; with ``deps=True``
(default) every transitive REQUIRES is added and re-run only when it is not up to date.
"""

from __future__ import annotations

import glob
import importlib
import json
import logging
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import normalize, params_hash
from .episode import Episode, read_meta
from .errors import PipelineError

log = logging.getLogger(__name__)

ORDER: tuple[str, ...] = (
    "ingest", "hands2d", "depth", "hand3d", "slam", "imu", "world",
    "arms", "objects", "scene", "retarget", "physics", "viz",
)
STEP_PACKAGE = "glassesdata_pipeline.steps"
RUNS_DIR = "_runs"


# =============================================================================================
# Registry
# =============================================================================================


@dataclass
class Step:
    """A step slot: the module (when implemented) and its declarations."""

    name: str
    module: Any = None
    status: str = "missing"  # "ok" | "missing" (not implemented yet) | "broken" (import failed)
    error: str = ""

    @classmethod
    def from_module(cls, name: str, module: Any) -> "Step":
        step = cls(name, module, "ok")
        problem = step._validate()
        if problem:
            step.status, step.error = "broken", problem
        return step

    def _validate(self) -> str:
        m = self.module
        if not callable(getattr(m, "run", None)):
            return "module has no run(ep, cfg)"
        declared = getattr(m, "NAME", self.name)
        if declared != self.name:
            return f"NAME = {declared!r} but the module is {self.name!r}"
        return ""

    @property
    def implemented(self) -> bool:
        return self.status == "ok"

    @property
    def version(self) -> Any:
        return getattr(self.module, "VERSION", 1)

    @property
    def requires(self) -> list[str]:
        return list(getattr(self.module, "REQUIRES", []) or [])

    @property
    def outputs(self) -> list[str]:
        return list(getattr(self.module, "OUTPUTS", []) or [])

    @property
    def config_sections(self) -> list[str]:
        return list(getattr(self.module, "CONFIG_SECTIONS", [self.name]) or [])

    def inputs(self, cfg: Mapping[str, Any]) -> list[str]:
        value = getattr(self.module, "INPUTS", []) or []
        if callable(value):
            value = value(cfg)
        return list(value)

    def params(self, cfg: Mapping[str, Any]) -> dict[str, Any]:
        return normalize({section: cfg.get(section, {}) for section in self.config_sections})


def discover(names: Sequence[str] = ORDER, package: str = STEP_PACKAGE) -> dict[str, Step]:
    """Import ``package.<name>`` for each name; never raises."""
    steps: dict[str, Step] = {}
    for name in names:
        module_name = f"{package}.{name}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name in (module_name, package) or (error.name and module_name.startswith(error.name + ".")):
                steps[name] = Step(name, None, "missing", "not implemented yet")
            else:
                steps[name] = Step(name, None, "broken", f"{type(error).__name__}: {error}")
            continue
        except Exception as error:  # noqa: BLE001 - a broken step must not break the CLI
            steps[name] = Step(name, None, "broken", f"{type(error).__name__}: {error}")
            continue
        steps[name] = Step.from_module(name, module)
    return steps


# =============================================================================================
# Cache
# =============================================================================================


def _stat(path: Path) -> list[int] | None:
    try:
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
            return [sum(p.stat().st_size for p in files), max((p.stat().st_mtime_ns for p in files), default=0)]
        st = path.stat()
        return [int(st.st_size), int(st.st_mtime_ns)]
    except FileNotFoundError:
        return None


def input_stats(ep: Episode, relpaths: Iterable[str]) -> dict[str, list[int] | None]:
    """``{relpath: [size, mtime_ns] or None}`` relative to the recording; globs are expanded."""
    stats: dict[str, list[int] | None] = {}
    for rel in relpaths:
        if any(ch in rel for ch in "*?["):
            matches = sorted(glob.glob(str(ep.path / rel), recursive=True))
            for match in matches:
                stats[str(Path(match).relative_to(ep.path))] = _stat(Path(match))
            if not matches:
                stats[rel] = None
        else:
            stats[rel] = _stat(ep.path / rel)
    return stats


def _record_path(ep: Episode, name: str) -> Path:
    return ep.path / "derived" / RUNS_DIR / f"{name}.json"


def cache_status(ep: Episode, step: Step, cfg: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    """``(up_to_date, reason)`` for an implemented step."""
    cfg = ep.config if cfg is None else cfg
    outputs = step.outputs
    if not outputs:
        return False, "the step declares no OUTPUTS"
    missing = [rel for rel in outputs if not (ep.path / "derived" / rel).exists()]
    if missing:
        return False, f"missing output {missing[0]}"
    expected = {
        "version": normalize(step.version),
        "params_hash": params_hash(step.params(cfg)),
        "inputs": normalize(input_stats(ep, step.inputs(cfg))),
    }
    metas = [m for m in (read_meta(ep.path / "derived" / rel) for rel in outputs) if m is not None]
    if not metas:
        record = _record_path(ep, step.name)
        if not record.is_file():
            return False, "no meta recorded"
        try:
            metas = [json.loads(record.read_text())["meta"]]
        except (OSError, ValueError, KeyError):
            return False, "unreadable run record"
    for meta in metas:
        if meta.get("step") not in (None, step.name):
            return False, f"output written by step {meta.get('step')!r}"
        if normalize(meta.get("version")) != expected["version"]:
            return False, f"version {meta.get('version')} -> {step.version}"
        if params_hash(meta.get("params", {})) != expected["params_hash"]:
            return False, "parameters changed"
        if normalize(meta.get("inputs", {})) != expected["inputs"]:
            changed = sorted(
                set(expected["inputs"]) ^ set(meta.get("inputs", {}))
                | {k for k in expected["inputs"] if meta.get("inputs", {}).get(k) != expected["inputs"][k]}
            )
            return False, f"inputs changed ({', '.join(changed[:3])})"
    return True, "up to date"


# =============================================================================================
# Running
# =============================================================================================


@dataclass
class StepResult:
    name: str
    status: str  # ran | cached | would-run | not-implemented | unavailable | blocked | failed | external
    seconds: float = 0.0
    summary: Any = None
    reason: str = ""


@dataclass
class RunReport:
    results: list[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(r.status == "failed" for r in self.results)

    def by_name(self) -> dict[str, StepResult]:
        return {r.name: r for r in self.results}

    def table(self) -> str:
        lines = [f"{'step':<10} {'status':<16} {'time':>8}  details"]
        for r in self.results:
            detail = r.reason or (_summary_text(r.summary) if r.summary else "")
            lines.append(f"{r.name:<10} {r.status:<16} {r.seconds:>7.2f}s  {detail}")
        return "\n".join(lines)


def _summary_text(summary: Any) -> str:
    if isinstance(summary, Mapping):
        return ", ".join(f"{k}={v}" for k, v in summary.items())
    return str(summary)


class Runner:
    """Runs steps on one :class:`Episode`.

    >>> report = Runner(ep).run(steps=["ingest"], force=True)
    >>> report.ok, report.table()
    """

    def __init__(self, ep: Episode, steps: Mapping[str, Step] | None = None, order: Sequence[str] = ORDER) -> None:
        self.ep = ep
        self.order = list(order)
        self.steps: dict[str, Step] = dict(steps) if steps is not None else discover(self.order)
        for name in self.order:
            self.steps.setdefault(name, Step(name, None, "missing", "not implemented yet"))

    # ---------------------------------------------------------------- selection

    def select(
        self,
        steps: Iterable[str] | None = None,
        from_: str | None = None,
        until: str | None = None,
    ) -> list[str]:
        """Explicitly selected steps in ORDER."""
        for name in [*(steps or []), from_, until]:
            if name is not None and name not in self.order:
                raise PipelineError(f"unknown step {name!r}; steps are: {', '.join(self.order)}")
        chosen = list(self.order)
        if steps:
            wanted = set(steps)
            chosen = [n for n in chosen if n in wanted]
        if from_ is not None:
            chosen = [n for n in chosen if self.order.index(n) >= self.order.index(from_)]
        if until is not None:
            chosen = [n for n in chosen if self.order.index(n) <= self.order.index(until)]
        return chosen

    def with_dependencies(self, names: Iterable[str]) -> list[str]:
        """``names`` plus every transitive REQUIRES, in ORDER."""
        needed: set[str] = set()
        stack = list(names)
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            step = self.steps.get(name)
            if step is not None and step.implemented:
                for req in step.requires:
                    if req not in self.steps:
                        raise PipelineError(f"step {name!r} requires unknown step {req!r}")
                    stack.append(req)
        return [n for n in self.order if n in needed]

    # ---------------------------------------------------------------- run

    def run(
        self,
        steps: Iterable[str] | None = None,
        from_: str | None = None,
        until: str | None = None,
        force: bool = False,
        deps: bool = True,
        dry_run: bool = False,
    ) -> RunReport:
        selected = self.select(steps, from_, until)
        plan = self.with_dependencies(selected) if deps else selected
        explicit = set(selected)
        report = RunReport()
        unavailable: dict[str, str] = {}  # step -> why its outputs cannot be relied on
        cfg = self.ep.config
        log.info("%s: steps %s", self.ep.path.name, " -> ".join(plan) if plan else "(none)")
        for name in plan:
            step = self.steps[name]
            result = self._run_one(step, cfg, force and name in explicit, unavailable, dry_run, plan)
            report.results.append(result)
            if result.status in ("failed", "blocked", "not-implemented", "unavailable"):
                unavailable[name] = result.status
        return report

    def _outputs_exist(self, step: Step) -> bool:
        return bool(step.outputs) and all((self.ep.path / "derived" / rel).exists() for rel in step.outputs)

    def _run_one(
        self, step: Step, cfg: Any, force: bool, unavailable: dict[str, str], dry_run: bool, plan: list[str]
    ) -> StepResult:
        name = step.name
        if not step.implemented:
            if self._outputs_exist(step):
                log.info("[%s] not implemented here; using its existing outputs", name)
                return StepResult(name, "external", reason="not implemented; existing outputs are used")
            if step.status == "missing":
                log.warning("[%s] not implemented yet (no module %s.%s)", name, STEP_PACKAGE, name)
                return StepResult(name, "not-implemented", reason="not implemented yet")
            log.error("[%s] unavailable: %s", name, step.error)
            return StepResult(name, "unavailable", reason=step.error)
        for req in step.requires:
            if req in unavailable:
                req_step = self.steps.get(req)
                if req_step is not None and not req_step.implemented and self._outputs_exist(req_step):
                    continue  # produced elsewhere; use what is there
                log.warning("[%s] blocked: requires %s (%s)", name, req, unavailable[req])
                return StepResult(name, "blocked", reason=f"requires {req} ({unavailable[req]})")
            if req not in plan:
                req_step = self.steps.get(req)
                outputs = req_step.outputs if req_step is not None else []
                if outputs and not all((self.ep.path / "derived" / rel).exists() for rel in outputs):
                    log.warning("[%s] blocked: requires %s, whose outputs are missing", name, req)
                    return StepResult(name, "blocked", reason=f"requires {req} (outputs missing)")
        up_to_date, reason = cache_status(self.ep, step, cfg)
        if up_to_date and not force:
            log.info("[%s] cached (%s)", name, reason)
            return StepResult(name, "cached", reason=reason)
        why = "forced" if force else reason
        if dry_run:
            log.info("[%s] would run (%s)", name, why)
            return StepResult(name, "would-run", reason=why)
        log.info("[%s] running (%s)", name, why)
        params = step.params(cfg)
        inputs = input_stats(self.ep, step.inputs(cfg))
        self.ep._context = {
            "step": name, "version": step.version, "params": params,
            "params_hash": params_hash(params), "inputs": inputs,
        }
        started = time.perf_counter()
        try:
            summary = step.module.run(self.ep, cfg)
        except Exception as error:  # noqa: BLE001
            seconds = time.perf_counter() - started
            log.error("[%s] failed after %.2f s: %s\n%s", name, seconds, error, traceback.format_exc())
            return StepResult(name, "failed", seconds, reason=f"{type(error).__name__}: {error}")
        finally:
            self.ep._context = None
        seconds = time.perf_counter() - started
        missing = [rel for rel in step.outputs if not (self.ep.path / "derived" / rel).exists()]
        if missing:
            log.error("[%s] finished but did not write %s", name, ", ".join(missing))
            return StepResult(name, "failed", seconds, summary, reason=f"did not write {missing[0]}")
        self._write_record(step, params, inputs, seconds, summary)
        log.info("[%s] done in %.2f s%s", name, seconds, f": {_summary_text(summary)}" if summary else "")
        return StepResult(name, "ran", seconds, summary)

    def _write_record(self, step: Step, params: dict, inputs: dict, seconds: float, summary: Any) -> None:
        record = {
            "meta": {
                "step": step.name, "version": step.version, "params": params,
                "params_hash": params_hash(params), "inputs": inputs,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "outputs": {rel: _stat(self.ep.path / "derived" / rel) for rel in step.outputs},
            "seconds": round(seconds, 3),
            "summary": summary,
        }
        path = _record_path(self.ep, step.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalize(record), indent=2))

    # ---------------------------------------------------------------- status

    def describe(self, with_cache: bool = True) -> list[dict[str, Any]]:
        """One row per step: name, status, version, requires, cache (when ``with_cache``)."""
        rows = []
        for name in self.order:
            step = self.steps[name]
            row: dict[str, Any] = {
                "name": name,
                "status": {"ok": "implemented", "missing": "not implemented yet", "broken": "unavailable"}[step.status],
                "version": step.version if step.implemented else None,
                "requires": step.requires if step.implemented else [],
                "error": step.error if step.status == "broken" else "",
            }
            if with_cache and step.implemented:
                row["cache"] = cache_status(self.ep, step)[1]
            rows.append(row)
        return rows


def describe_steps(steps: Mapping[str, Step] | None = None, order: Sequence[str] = ORDER) -> list[dict[str, Any]]:
    """Registry rows without a recording (for ``list-steps``)."""
    steps = dict(steps) if steps is not None else discover(order)
    rows = []
    for name in order:
        step = steps.get(name, Step(name))
        rows.append({
            "name": name,
            "status": {"ok": "implemented", "missing": "not implemented yet", "broken": "unavailable"}[step.status],
            "version": step.version if step.implemented else None,
            "requires": step.requires if step.implemented else [],
            "outputs": step.outputs if step.implemented else [],
            "error": step.error if step.status == "broken" else "",
        })
    return rows


def run_steps(
    path: str | Path,
    steps: Iterable[str] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> RunReport:
    """Convenience: ``Runner(Episode(path, config_overrides)).run(steps, **kwargs)``."""
    return Runner(Episode(path, config_overrides)).run(steps, **kwargs)
