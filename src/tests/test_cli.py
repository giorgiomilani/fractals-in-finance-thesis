import subprocess, sys, importlib.util

def test_cli_runs():
    code = subprocess.call([sys.executable, "-m", "fractalfinance.cli", "--help"])
    assert code == 0
