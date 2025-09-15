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


def test_plot_cli(tmp_path):
    out = tmp_path / "fbm.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fractalfinance.cli",
            "plot",
            "fbm",
            "--path",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out.exists()

