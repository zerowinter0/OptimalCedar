#!/usr/bin/env bash
set -euo pipefail

cd /workspace/OptimalCedar
source env/bin/activate

python - <<'PY'
import cedar
import numpy
import pyarrow
import ray
import tensorflow as tf
import torch
import torchdata
import torchtext
import torchvision

print("cedar import ok")
print(f"python env ok")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torchdata={torchdata.__version__}")
print(f"torchtext={torchtext.__version__}")
print(f"tensorflow={tf.__version__}")
print(f"ray={ray.__version__}")
print(f"numpy={numpy.__version__}")
print(f"pyarrow={pyarrow.__version__}")
PY

pytest -q \
  tests/test_reorder.py \
  tests/test_dp_cache_fusion_optimizer.py \
  tests/test_sources.py::test_basic_itersource
