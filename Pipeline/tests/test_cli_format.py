import ast
import json
from pathlib import Path

import pytest

from glassesdata_pipeline import recording
from glassesdata_pipeline.__main__ import main

RECEIVER = Path(__file__).resolve().parents[2] / "Receiver" / "glassesdata_receiver" / "recorder.py"


def test_csv_headers_match_the_receiver():
    if not RECEIVER.is_file():
        pytest.skip("Receiver/ is not next to Pipeline/")
    tree = ast.parse(RECEIVER.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id.endswith("_HEADER"):
            name = node.targets[0].id
            if name == "HANDS_HEADER":
                continue  # built with comprehensions; compared through its length below
            found[name] = ast.literal_eval(node.value)
    for name, value in found.items():
        assert getattr(recording, name) == value, name
    assert len(recording.HANDS_HEADER) == 5 + 2 * 63


def test_cli_list_run_eval_and_lazy(synth_recording, tmp_path, capsys):
    assert main(["list-steps"]) == 0
    out = capsys.readouterr().out
    assert "ingest" in out and "implemented" in out  # every step module exists now
    rec = tmp_path / "rec"
    import shutil

    shutil.copytree(synth_recording, rec)
    assert main(["run", str(rec), "--steps", "ingest"]) == 0
    assert (rec / "derived" / "timeline.npz").is_file()
    assert main(["run", str(rec), "--steps", "ingest", "--set", "ingest.imu_gap_warn_s=0.2"]) == 0
    assert "ran" in capsys.readouterr().out  # a changed parameter re-runs the step
    assert main(["list-steps", str(rec), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "ingest" and rows[0]["cache"] == "parameters changed"
    assert main(["eval", str(rec), "--no-plots"]) == 0
    assert (rec / "derived" / "eval.json").is_file()
    assert main(["view", str(rec)]) == 2  # no derived/viz/episode.rrd yet: says how to create / open one
    assert "episode.rrd" in capsys.readouterr().out
    import importlib.util

    if importlib.util.find_spec("rsl_rl") is None:  # train is implemented (rl/); without rsl_rl it names the install
        assert main(["train", str(rec)]) == 1
        assert "rsl-rl-lib" in capsys.readouterr().err
    assert main(["eval-charuco", str(rec)]) == 1  # implemented; this recording has no board definition
    assert main(["run", str(rec), "--steps", "nope"]) == 1


def test_cli_synth(tmp_path):
    out = tmp_path / "tiny"
    assert main(["synth", str(out), "--seconds", "1", "--size", "90x160", "--seed", "3"]) == 0
    info = json.loads((out / "info.json").read_text())
    assert info["synth"]["seed"] == 3 and info["framesWritten"] == 24
    assert main(["synth", str(out), "--seconds", "1", "--size", "90x160"]) == 1  # not empty
