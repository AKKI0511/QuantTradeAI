import os
import subprocess
import sys


def test_python_module_cli_help_smoke():
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    result = subprocess.run(
        [sys.executable, "-m", "quanttradeai.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "QuantTradeAI command line interface" in result.stdout
    assert "train" not in result.stdout
    assert "backtest-model" not in result.stdout
    assert "live-trade" not in result.stdout
    assert "validate-config" not in result.stdout
