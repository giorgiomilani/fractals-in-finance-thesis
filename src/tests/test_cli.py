import importlib.util
import subprocess
import sys


def test_cli_runs():
    result = subprocess.run(
        [sys.executable, "-m", "fractalfinance.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
