import subprocess
import sys


def test_cpu_pipe_import_does_not_eagerly_load_tensorflow():
    script = (
        "import sys; import cedar.pipes; "
        "assert 'tensorflow' not in sys.modules, "
        "sorted(k for k in sys.modules if k.startswith('tensorflow'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
