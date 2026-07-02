import os
import subprocess
import sys
from pathlib import Path


def test_train_lightning_help_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path)}

    result = subprocess.run(
        [sys.executable, "scripts/train_lightning.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
