ARG BUILDER_IMAGE=optimalcedar-tensorflow-builder:plumber-08bf144ec13b
FROM ${BUILDER_IMAGE} AS tensorflow_builder

FROM tensorflow/tensorflow:2.7.0-gpu

ARG PLUMBER_APP_ARCHIVE=plumber-app-6123f5bce36e.tar.gz
ARG PLUMBER_APP_SHA256=e799ab4b559361e4f5dac1950a2d207b0b3afc21386d1ae04e1555f5aeed9a34

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

RUN python -m pip uninstall -y \
      keras \
      tensorboard \
      tensorflow-estimator \
    && python -m pip install --no-cache-dir \
      flatbuffers==1.12 \
      h5py==3.1.0 \
      keras-nightly==2.6.0.dev2021052700 \
      libclang==11.1.0 \
      six==1.15.0 \
      tensorboard==2.5.0 \
      tensorflow-estimator==2.5.0 \
      typing-extensions==3.7.4.3 \
      wrapt==1.12.1

RUN python -m pip install --no-cache-dir \
      cvxpy==1.3.2 \
      ecos==2.0.12 \
      huggingface-hub==0.16.4 \
      librosa==0.9.2 \
      llvmlite==0.39.1 \
      matplotlib==3.7.5 \
      networkx==2.8.8 \
      numba==0.56.4 \
      numpy==1.21.6 \
      osqp==0.6.3 \
      packaging==23.2 \
      pandas==1.5.3 \
      Pillow==9.5.0 \
      psutil==5.9.8 \
      pydot==1.4.2 \
      PyYAML==6.0.1 \
      qdldl==0.1.7.post0 \
      scipy==1.10.1 \
      scs==3.2.2 \
      tensorflow-addons==0.15.0 \
      tokenizers==0.13.3 \
      transformers==4.30.2 \
      typeguard==2.13.3

COPY sources/${PLUMBER_APP_ARCHIVE} /tmp/plumber-app.tar.gz
RUN echo "${PLUMBER_APP_SHA256}  /tmp/plumber-app.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/plumber-app \
    && tar -xzf /tmp/plumber-app.tar.gz \
      --strip-components=1 -C /opt/plumber-app \
    && rm /tmp/plumber-app.tar.gz \
    && python -m pip install --no-cache-dir \
      nvidia-pyindex==1.0.9 \
    && python -m pip install --no-cache-dir graphsurgeon \
    && python -m pip install --no-cache-dir --no-deps \
      /opt/plumber-app/plumber_analysis

WORKDIR /workspace/OptimalCedar
CMD ["sleep", "infinity"]
