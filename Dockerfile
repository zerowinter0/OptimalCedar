FROM pytorch/pytorch:latest

ARG HTTP_PROXY=http://127.0.0.1:7891
ARG HTTPS_PROXY=http://127.0.0.1:7891
ARG NO_PROXY=localhost,127.0.0.1
ARG http_proxy=http://127.0.0.1:7891
ARG https_proxy=http://127.0.0.1:7891
ARG no_proxy=localhost,127.0.0.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    NO_PROXY=$NO_PROXY \
    http_proxy=$http_proxy \
    https_proxy=$https_proxy \
    no_proxy=$no_proxy

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace/OptimalCedar

COPY docker/requirements-docker.txt /tmp/requirements-docker.txt

RUN python -m venv --system-site-packages env \
    && source env/bin/activate \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements-docker.txt \
    && pip install --no-deps torchvision==0.15.2

COPY . .

RUN source env/bin/activate \
    && pip install -e . \
    && python - <<'PY'
import cedar
import ray
import tensorflow as tf
import torch

print("OptimalCedar container build check")
print(f"torch={torch.__version__}")
print(f"tensorflow={tf.__version__}")
print(f"ray={ray.__version__}")
PY

ENV PATH="/workspace/OptimalCedar/env/bin:${PATH}" \
    PYTHONPATH="/workspace/OptimalCedar"

CMD ["bash"]
