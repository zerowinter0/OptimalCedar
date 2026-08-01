FROM datajuicer-cuda12.2:latest

ARG DATAJUICER_COMMIT=bb3d88aac183cc22b6f816262a812a9e5d5abb57
LABEL org.opencontainers.image.title="OptimalCedar pinned Data-Juicer baseline" \
      org.opencontainers.image.revision="${DATAJUICER_COMMIT}" \
      org.opencontainers.image.base.name="datajuicer-cuda12.2:latest"

COPY data-juicer /opt/data-juicer
WORKDIR /opt/data-juicer
ENV PYTHONPATH=/opt/data-juicer

# The local CUDA 12.2 Data-Juicer image supplies the pinned system and model
# dependencies. PYTHONPATH overlays the exact experiment checkout on that
# environment; validate that both Python and the CLI resolve the pinned source.
RUN uv pip install --system \
        'ray[default]==2.52.0' \
        'sentencepiece==0.2.0' \
        'fasttext-wheel==0.9.2' \
        'kenlm==0.3.0' \
        'ftfy==6.3.1' \
    && python3 -m pip check \
    && python3 -c 'import pathlib, data_juicer; assert pathlib.Path(data_juicer.__file__).is_relative_to("/opt/data-juicer")' \
    && dj-process --help >/dev/null

CMD ["tail", "-f", "/dev/null"]
