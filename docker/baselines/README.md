# Plumber and FastFlow container sources

Both systems require a modified TensorFlow 2.7 runtime. Stock TensorFlow is
not a valid substitute.

Pinned sources:

- Plumber TensorFlow:
  `mkuchnik/PlumberTensorflow@08bf144ec13b0c27f2a02aaba975546506ee0f6a`
- Plumber application:
  `mkuchnik/PlumberApp@6123f5bce36eec7dc75b6b9298054b493d930bdc`
- FastFlow TensorFlow Software Heritage revision:
  `swh:1:rev:8dc4caf647dc2df3c9fbe6c15bc7377baa8db1d6`
- FastFlow TensorFlow directory:
  `swh:1:dir:8fe31ad60626b9bcd9ce2917c8458e8fdec0f1e8`
- FastFlow application Software Heritage revision:
  `swh:1:rev:f2e3a3363e9535fcd39b93ccc69c91f7a07e5ea6`
- FastFlow application directory:
  `swh:1:dir:85cbfadf61d642f3c4fd914fd84564182a1b5c45`

The FastFlow GitHub repositories referenced by the paper are no longer
publicly available. Software Heritage snapshots are the preserved author
sources, not a reimplementation.

The builder targets CUDA 11.2/cuDNN 8 and compute capability 8.6 for the RTX
A6000. Runtime containers receive the same GPU set, host network, 64 GiB
shared memory, all 64 host CPUs, and the same writable OptimalCedar bind mount
as `optimalcedar-torch201-dev`.

The host VPN proxy is exposed on `127.0.0.1:17890`. Builds use host networking
and explicit proxy build arguments, while the host-networked runtime
containers receive matching upper- and lower-case proxy environment
variables. Override `VPN_HTTP_PROXY`, `VPN_ALL_PROXY`, or `VPN_NO_PROXY` when
the host VPN endpoint changes.

## Background FastFlow build

Once the pinned source archives are present, the remaining FastFlow build can
be run in the background:

```bash
mkdir -p evaluation/chapter6_experiments/container_setup_logs
nohup bash docker/baselines/finish_fastflow_setup.sh \
  >evaluation/chapter6_experiments/container_setup_logs/fastflow_setup.log \
  2>&1 </dev/null &
echo $! >evaluation/chapter6_experiments/container_setup_logs/fastflow_setup.pid
```

The machine-readable status is written to
`evaluation/chapter6_experiments/container_setup_logs/fastflow_setup.status`.
`state=SUCCEEDED` means that the image was built, the container was started,
and CPU, GPU, shared-memory, repository-mount, FastFlow import, and custom
TensorFlow API checks all passed.
