"""Command-line arguments that name a port or an address are checked before anything starts."""

import threading

import pytest

from glassesdata_receiver.__main__ import parse_args
from glassesdata_receiver.link import UsbSource
from glassesdata_receiver.receiver import Receiver


@pytest.mark.parametrize("argv, message", [
    (["--port", "70000"], "is not a port"), (["--port", "-1"], "is not a port"), (["--port", "seven"], "is not a port number"),
    (["--port", "0"], "only works for the --wifi listener"), (["--usb", "--wifi", "--port", "0"], "only works for the --wifi listener"),
    (["--tcp", "127.0.0.1"], "is not HOST:PORT"), (["--tcp", "127.0.0.1:99999"], "is not a port"), (["--tcp", "host:x"], "is not a port number"),
    (["--tcp", "host:0"], "is not a port"),
])
def test_bad_ports_and_addresses_are_argument_errors(argv, message, capsys):
    with pytest.raises(SystemExit) as stopped:
        parse_args(argv)
    assert stopped.value.code == 2 and message in capsys.readouterr().err  # argparse's usage error, no traceback


@pytest.mark.parametrize("text, address", [("127.0.0.1:7901", ("127.0.0.1", 7901)), (":7901", ("127.0.0.1", 7901)),
                                           ("[::1]:7901", ("::1", 7901)), ("laptop.local:80", ("laptop.local", 80))])
def test_tcp_addresses(text, address):
    assert parse_args(["--tcp", text]).tcp == address


def test_good_ports():
    assert parse_args(["--wifi", "--port", "0"]).port == 0  # any free port for the Wi-Fi listener
    assert parse_args(["--port", "7900"]).port == 7900 and parse_args([]).port == 7900


def test_a_usb_source_with_a_bad_port_logs_and_stops_cleanly(tmp_path, caplog):
    hub = Receiver(tmp_path)
    try:
        for port in (70000, 0, "7900"):
            source = UsbSource(hub, port, threading.Event())
            with caplog.at_level("ERROR"):
                source.run()  # returns at once instead of dying in the thread
            assert "is not a port" in hub.status().sources["usb"]
        assert any("USB link not started" in record.getMessage() for record in caplog.records)
    finally:
        hub.shutdown()
