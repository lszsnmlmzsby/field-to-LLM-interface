"""Public commands resolve to the shared workflow implementations."""
import pytest
import field


@pytest.mark.parametrize("command", field.COMMANDS)
def test_command_help(command, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["field.py"])
    with pytest.raises(SystemExit) as result:
        field.main([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out.lower()


def test_evaluate_selects_read_only_mode(monkeypatch):
    import sys
    seen = []
    class Workflow:
        def main(self):
            seen.extend(sys.argv)
    monkeypatch.setattr(field, "import_module", lambda name: Workflow())
    monkeypatch.setattr(sys, "argv", ["field.py"])
    field.main(["evaluate", "--resume", "best.pt"])
    assert seen == ["field.py evaluate", "--resume", "best.pt", "--evaluate-only"]


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit, match="Unknown command"):
        field.main(["missing"])
