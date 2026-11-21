import pytest

from glassesdata_pipeline.config import Config, load_config, overrides_from_cli, parse_set, parse_value
from glassesdata_pipeline.errors import ConfigError


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "configs"
    d.mkdir()
    (d / "alpha.toml").write_text('[alpha]\na = 1\nb = "x"\n[alpha.sub]\nc = 3\nd = [1, 2]\n')
    (d / "beta.toml").write_text("[beta]\nflag = false\nratio = 0.5\n")
    return d


def test_merges_every_file_in_the_folder(config_dir):
    cfg = load_config(directory=config_dir)
    assert cfg["alpha"]["a"] == 1 and cfg["beta"]["ratio"] == 0.5
    (config_dir / "gamma.toml").write_text("[gamma]\nnew = 7\n")
    assert load_config(directory=config_dir).get("gamma.new") == 7  # no fixed list of files


def test_precedence_recording_then_file_then_set(config_dir, tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "pipeline.toml").write_text('[alpha]\na = 2\n[alpha.sub]\nc = 30\n[unknown]\nkey = "kept"\n')
    extra = tmp_path / "extra.toml"
    extra.write_text("[alpha]\na = 3\n[beta]\nflag = true\n")
    cfg = load_config(recording=rec, config_file=extra, sets=["alpha.a=4", "alpha.sub.d=[9]"], directory=config_dir)
    assert cfg["alpha"]["a"] == 4  # --set wins
    assert cfg["beta"]["flag"] is True  # --config over defaults
    assert cfg["alpha"]["sub"]["c"] == 30  # recording over defaults, nested merge
    assert cfg["alpha"]["sub"]["d"] == [9]
    assert cfg["alpha"]["b"] == "x"  # untouched default survives the merge
    assert cfg["unknown"]["key"] == "kept"  # unknown keys are allowed
    cfg2 = load_config(recording=rec, config_file=extra, directory=config_dir)
    assert cfg2["alpha"]["a"] == 3  # --config over recording


def test_extra_dict_is_highest(config_dir):
    cfg = load_config(sets=["alpha.a=5"], extra={"alpha": {"a": 6}}, directory=config_dir)
    assert cfg.alpha.a == 6


def test_dotted_access(config_dir):
    cfg = load_config(directory=config_dir)
    assert cfg.get("alpha.sub.c") == 3
    assert cfg["alpha.sub.c"] == 3
    assert "alpha.sub.c" in cfg and "alpha.nope" not in cfg
    assert cfg.get("alpha.nope", 42) == 42
    assert cfg.alpha.sub.c == 3
    assert isinstance(cfg.alpha, Config)
    assert cfg.section("missing") == {}
    with pytest.raises(AttributeError):
        _ = cfg.missing
    with pytest.raises(KeyError):
        _ = cfg["alpha.nope"]
    cfg.set("delta.x.y", 1)
    assert cfg["delta"]["x"]["y"] == 1
    assert cfg.to_dict()["alpha"]["sub"] == {"c": 3, "d": [1, 2]}


def test_parse_values():
    assert parse_value("2") == 2
    assert parse_value("2.5") == 2.5
    assert parse_value("true") is True
    assert parse_value('"quoted"') == "quoted"
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("{a = 1}") == {"a": 1}
    assert parse_value("gt") == "gt"  # plain string fallback
    assert parse_value("some/path.txt") == "some/path.txt"
    assert parse_set("hands2d.backend=gt") == {"hands2d": {"backend": "gt"}}
    assert parse_set("a.b.c = 1") == {"a": {"b": {"c": 1}}}
    with pytest.raises(ConfigError):
        parse_set("no_equals")
    with pytest.raises(ConfigError):
        parse_set(".x=1")


def test_overrides_from_cli(tmp_path):
    f = tmp_path / "o.toml"
    f.write_text("[s]\na = 1\nb = 2\n")
    assert overrides_from_cli(f, ["s.b=3"]) == {"s": {"a": 1, "b": 3}}
    assert overrides_from_cli(None, []) == {}


def test_bad_toml_names_the_file(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    (d / "bad.toml").write_text("[x\n")
    with pytest.raises(ConfigError, match="bad.toml"):
        load_config(directory=d)


def test_shipped_configs_load():
    cfg = load_config()
    for section in ("camera", "ingest", "synth", "eval"):
        assert section in cfg, section
    assert cfg.get("synth.gt_noise.hands2d_sigma_px") == 2.0
    assert cfg.get("synth.imu.time_offset_s") == 0.02


def test_heavy_extras_match_requirements():
    """pyproject.toml's [heavy] extras carry the same pins as requirements.txt (the Linux install)."""
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    req = {}
    for line in (root / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-r"):
            req[re.split(r"[<>=!~]", line)[0]] = line
    heavy = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]["heavy"]
    assert {re.split(r"[<>=!~]", spec)[0]: spec for spec in heavy} == req
