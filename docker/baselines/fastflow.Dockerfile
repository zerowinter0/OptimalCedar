ARG BUILDER_IMAGE=optimalcedar-tensorflow-builder:fastflow-8dc4caf647dc
FROM ${BUILDER_IMAGE} AS tensorflow_builder

FROM tensorflow/tensorflow:2.7.0-gpu

ARG FASTFLOW_ARCHIVE=fastflow-f2e3a3363e95.tar.gz
ARG FASTFLOW_ARCHIVE_SHA256=a2d2281377130074586384b3d4ea2e61e6f5d777e055d67c344b249d54e7cd3a

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir \
      pip==23.3.2 \
      setuptools==68.2.2 \
      wheel==0.41.3

RUN curl -fsSL \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
      | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tensorflow_builder /wheel/ /tmp/tensorflow-wheel/
RUN python -m pip install --no-cache-dir --force-reinstall --no-deps \
      /tmp/tensorflow-wheel/tensorflow-*.whl \
    && rm -rf /tmp/tensorflow-wheel

RUN python -m pip install --no-cache-dir --retries 10 --timeout 60 \
      boto3==1.24.76 \
      grpcio-tools==1.44.0 \
      librosa==0.9.2 \
      paramiko==2.11.0 \
      Pillow==9.5.0 \
      protobuf==3.19.6 \
      psutil==5.9.2 \
      PyYAML==6.0 \
      requests==2.28.1 \
      scp==0.14.4 \
      tensorflow-addons==0.15.0 \
      transformers==4.30.2

COPY sources/${FASTFLOW_ARCHIVE} /tmp/fastflow.tar.gz
RUN echo "${FASTFLOW_ARCHIVE_SHA256}  /tmp/fastflow.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/fastflow \
    && tar -xzf /tmp/fastflow.tar.gz \
      --strip-components=1 -C /opt/fastflow \
    && rm /tmp/fastflow.tar.gz \
    && sed -i '/fastflow-tensorflow==2.7.0/d' \
      /opt/fastflow/requirements.txt \
    && python -m pip install --no-cache-dir --no-deps /opt/fastflow

WORKDIR /workspace/OptimalCedar
CMD ["sleep", "infinity"]
