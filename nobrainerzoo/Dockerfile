FROM continuumio/miniconda3:4.12.0
  
RUN conda install -c conda-forge git-annex \
    && sync && conda clean --all && sync
  
RUN pip install --no-cache-dir datalad datalad-osf PyYAML chardet==4.0.0 \
    && rm -rf ~/.cache/pip/* && sync

RUN git config --global --add user.name "nobrainerzoo" \
  && git config --global --add user.email "nobrainerzoo"

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR "/work"
LABEL maintainer="Hoda Rajaei <rajaei.hoda@gmail.com>"