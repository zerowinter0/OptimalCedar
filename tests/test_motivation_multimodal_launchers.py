import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = tuple(
    ROOT / "evaluation" / "motivation_multimodal" / name
    for name in ("run_pilot.sh", "run_formal.sh", "run_scaling.sh")
)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_has_valid_shell_syntax(launcher: Path) -> None:
    subprocess.run(["bash", "-n", str(launcher)], check=True)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_records_fixed_resource_protocol(launcher: Path) -> None:
    text = launcher.read_text(encoding="utf-8")

    assert "env/bin/activate" in text
    assert "W=1" in text
    assert "CPU_BUDGET=64" in text
    assert 'PYTHONPATH="$REPO_ROOT' in text
    assert "kill -0" in text
    assert "run.pid" in text
    assert "run.log" in text
    assert "fallback" not in text.lower()
    assert "--max-configs" not in text
