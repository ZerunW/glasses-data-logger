import os
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from glassesdata_pipeline.episode import Episode
from glassesdata_pipeline.errors import PipelineError
from glassesdata_pipeline.runner import Runner, Step, cache_status, discover


def fake_step(name, calls, requires=(), inputs=(), outputs=None, version=1, fail=False, write=True, sections=None):
    def run(ep, cfg):
        calls.append(name)
        if fail:
            raise RuntimeError("boom")
        if write:
            for out in outputs or [f"{name}.npz"]:
                if out.endswith(".npz"):
                    ep.save_npz(out, ep.step_meta(), value=np.array([cfg.get(f"{name}.value", 0)]))
                else:
                    np.save(ep.file(out), np.zeros(3))
        return {"n": 1}

    module = SimpleNamespace(
        NAME=name, VERSION=version, REQUIRES=list(requires), INPUTS=list(inputs),
        OUTPUTS=list(outputs or [f"{name}.npz"]), CONFIG_SECTIONS=sections or [name], run=run,
    )
    return Step.from_module(name, module)


@pytest.fixture
def rec(tmp_path):
    path = tmp_path / "rec"
    path.mkdir()
    (path / "raw.txt").write_text("raw")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "x.toml").write_text("[a]\nvalue = 1\n[b]\nvalue = 1\n")
    return path


def episode(rec, overrides=None):
    return Episode(rec, overrides, config_dir=rec.parent / "configs")


def chain(calls, **kw):
    return {
        "a": fake_step("a", calls, inputs=["raw.txt"]),
        "b": fake_step("b", calls, requires=["a"], inputs=["derived/a.npz"], **kw.get("b", {})),
        "c": fake_step("c", calls, requires=["b"], inputs=["derived/b.npz"]),
    }


def statuses(report):
    return {r.name: r.status for r in report.results}


def test_runs_in_order_then_caches(rec):
    calls = []
    steps = chain(calls)
    report = Runner(episode(rec), steps, order=["a", "b", "c"]).run()
    assert calls == ["a", "b", "c"] and report.ok
    assert statuses(report) == {"a": "ran", "b": "ran", "c": "ran"}
    meta = Episode(rec).load_npz("b")["meta"]
    assert meta["step"] == "b" and meta["version"] == 1 and meta["params"] == {"b": {"value": 1}}
    assert list(meta["inputs"]) == ["derived/a.npz"]
    report = Runner(episode(rec), steps, order=["a", "b", "c"]).run()
    assert calls == ["a", "b", "c"]
    assert set(statuses(report).values()) == {"cached"}
    assert (rec / "derived" / "_runs" / "b.json").is_file()


def test_param_change_reruns_step_and_dependents(rec):
    calls = []
    steps = chain(calls)
    Runner(episode(rec), steps, order=["a", "b", "c"]).run()
    calls.clear()
    report = Runner(episode(rec, {"b": {"value": 2}}), steps, order=["a", "b", "c"]).run()
    assert calls == ["b", "c"]
    assert statuses(report)["a"] == "cached"
    assert Episode(rec).load_npz("b")["value"][0] == 2


def test_input_change_and_version_bump(rec):
    calls = []
    steps = chain(calls)
    Runner(episode(rec), steps, order=["a", "b", "c"]).run()
    calls.clear()
    (rec / "raw.txt").write_text("changed!")
    Runner(episode(rec), steps, order=["a", "b", "c"]).run()
    assert calls == ["a", "b", "c"]
    calls.clear()
    steps["c"] = fake_step("c", calls, requires=["b"], inputs=["derived/b.npz"], version=2)
    Runner(episode(rec), steps, order=["a", "b", "c"]).run()
    assert calls == ["c"]
    ok, reason = cache_status(episode(rec), steps["c"])
    assert ok and reason == "up to date"


def test_force_selection_and_dependencies(rec):
    calls = []
    steps = chain(calls)
    runner = Runner(episode(rec), steps, order=["a", "b", "c"])
    runner.run(steps=["b"])  # a is added as a dependency
    assert calls == ["a", "b"]
    calls.clear()
    report = runner.run(steps=["b"], force=True)
    assert calls == ["b"] and statuses(report) == {"a": "cached", "b": "ran"}
    calls.clear()
    runner.run(from_="b", force=True)
    assert calls == ["b", "c"]
    calls.clear()
    runner.run(until="b", force=True)
    assert calls == ["a", "b"]
    assert runner.select(from_="b", until="b") == ["b"]
    with pytest.raises(PipelineError, match="unknown step"):
        runner.select(steps=["nope"])


def test_no_deps_blocks_when_inputs_are_missing(rec):
    calls = []
    report = Runner(episode(rec), chain(calls), order=["a", "b", "c"]).run(steps=["c"], deps=False)
    assert calls == [] and statuses(report) == {"c": "blocked"}


def test_not_implemented_steps_are_reported_and_block_dependents(rec):
    calls = []
    steps = {"a": fake_step("a", calls, inputs=["raw.txt"]),
             "c": fake_step("c", calls, requires=["x"], inputs=["derived/x.npz"])}
    runner = Runner(episode(rec), steps, order=["a", "x", "c"])
    report = runner.run()
    assert statuses(report) == {"a": "ran", "x": "not-implemented", "c": "blocked"}
    assert report.ok  # nothing failed
    assert runner.describe()[1]["status"] == "not implemented yet"
    # outputs produced elsewhere (e.g. an imported trajectory) unblock the dependent
    Episode(rec).save_npz("x", {"step": "x"}, value=np.zeros(1))
    runner.steps["x"].module = SimpleNamespace(OUTPUTS=["x.npz"])  # declared outputs, still not implemented
    report = runner.run()
    assert statuses(report) == {"a": "cached", "x": "external", "c": "ran"}


def test_failures_block_dependents_only(rec):
    calls = []
    steps = chain(calls, b={"fail": True})
    steps["d"] = fake_step("d", calls, requires=["a"], inputs=["derived/a.npz"])
    report = Runner(episode(rec), steps, order=["a", "b", "c", "d"]).run()
    assert statuses(report) == {"a": "ran", "b": "failed", "c": "blocked", "d": "ran"}
    assert not report.ok
    assert "boom" in report.by_name()["b"].reason


def test_missing_outputs_fail_and_sidecar_cache(rec):
    calls = []
    steps = {"a": fake_step("a", calls, write=False)}
    report = Runner(episode(rec), steps, order=["a"]).run()
    assert statuses(report) == {"a": "failed"}
    steps = {"n": fake_step("n", calls, outputs=["n.npy"], inputs=["raw.txt"])}  # no meta-carrying output
    Runner(episode(rec), steps, order=["n"]).run()
    report = Runner(episode(rec), steps, order=["n"]).run()
    assert statuses(report) == {"n": "cached"}
    os.utime(rec / "raw.txt", ns=(1, 1))
    report = Runner(episode(rec), steps, order=["n"]).run()
    assert statuses(report) == {"n": "ran"}


def test_dry_run_runs_nothing(rec):
    calls = []
    report = Runner(episode(rec), chain(calls), order=["a", "b", "c"]).run(dry_run=True)
    assert calls == [] and statuses(report)["a"] == "would-run"


def test_discover_reports_missing_and_broken_modules(tmp_path, monkeypatch):
    pkg = tmp_path / "fakesteps_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text(textwrap.dedent("""
        NAME = "good"
        VERSION = 3
        REQUIRES = []
        OUTPUTS = ["good.npz"]
        def run(ep, cfg):
            return {}
    """))
    (pkg / "broken.py").write_text("import a_package_that_does_not_exist_123\n")
    (pkg / "wrong.py").write_text("NAME = 'other'\ndef run(ep, cfg):\n    return {}\n")
    (pkg / "norun.py").write_text("NAME = 'norun'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    steps = discover(["good", "absent", "broken", "wrong", "norun"], package="fakesteps_pkg")
    assert steps["good"].implemented and steps["good"].version == 3
    assert steps["absent"].status == "missing"
    assert steps["broken"].status == "broken" and "a_package_that_does_not_exist_123" in steps["broken"].error
    assert steps["wrong"].status == "broken" and "NAME" in steps["wrong"].error
    assert steps["norun"].status == "broken"


def test_real_registry_has_ingest():
    steps = discover()
    assert steps["ingest"].implemented
    assert all(name in steps for name in ("hands2d", "slam", "viz"))
