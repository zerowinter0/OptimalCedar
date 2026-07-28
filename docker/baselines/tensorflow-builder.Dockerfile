FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 AS tensorflow_builder

ARG TF_ARCHIVE
ARG TF_ARCHIVE_SHA256

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_BIN_PATH=/usr/bin/python3 \
    PYTHON_LIB_PATH=/usr/local/lib/python3.8/dist-packages \
    TF_NEED_OPENCL_SYCL=0 \
    TF_NEED_ROCM=0 \
    TF_NEED_CUDA=1 \
    TF_NEED_TENSORRT=0 \
    TF_CUDA_VERSION=11.2 \
    TF_CUDNN_VERSION=8 \
    TF_CUDA_PATHS=/usr/local/cuda,/usr \
    TF_CUDA_COMPUTE_CAPABILITIES=8.6 \
    GCC_HOST_COMPILER_PATH=/usr/bin/gcc \
    CC_OPT_FLAGS=-march=haswell \
    TF_SET_ANDROID_WORKSPACE=0 \
    USE_DEFAULT_PYTHON_LIB_PATH=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      openjdk-11-jdk-headless \
      patchelf \
      python3 \
      python3-dev \
      python3-pip \
      python-is-python3 \
      python3-setuptools \
      python3-wheel \
      rsync \
      unzip \
      zip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
      keras_preprocessing==1.1.2 \
      numpy==1.21.6 \
      packaging==23.2 \
      setuptools==59.6.0 \
      six==1.16.0 \
      wheel==0.37.1

COPY sources/bazel-3.7.2-linux-x86_64 /usr/local/bin/bazel
RUN chmod 0755 /usr/local/bin/bazel && bazel --version

COPY sources/${TF_ARCHIVE} /tmp/tensorflow.tar.gz
RUN echo "${TF_ARCHIVE_SHA256}  /tmp/tensorflow.tar.gz" | sha256sum -c -

RUN mkdir -p /src/tensorflow /wheel \
    && tar -xzf /tmp/tensorflow.tar.gz \
      --strip-components=1 -C /src/tensorflow \
    && rm /tmp/tensorflow.tar.gz

WORKDIR /src/tensorflow
RUN python3 configure.py
RUN --mount=type=cache,id=optimalcedar-tf27-bazel,target=/root/.cache/bazel \
    bazel \
      --output_user_root=/root/.cache/bazel \
      build \
      --config=opt \
      --config=cuda \
      --jobs=64 \
      --local_ram_resources=524288 \
      //tensorflow/tools/pip_package:build_pip_package \
    && ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /wheel
