import sys

import main as cli


def test_cli_writes_schedule(tmp_path, monkeypatch, capsys):
    out = tmp_path / "schedule.csv"
    monkeypatch.setattr(sys, "argv", ["main.py", "--hours", "24", "--out", str(out)])
    assert cli.main() == 0
    stdout = capsys.readouterr().out
    assert "Status: OPTIMAL" in stdout and "Value vs baselines" in stdout
    assert out.exists() and len(out.read_text().splitlines()) == 25


def test_cli_unknown_start(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--start", "2030-01-01"])
    assert cli.main() == 2
