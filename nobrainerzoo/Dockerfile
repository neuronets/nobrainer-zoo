FROM python:3.9.13-bullseye

RUN curl -sSL http://neuro.debian.net/lists/bionic.us-nh.full | tee /etc/apt/sources.list.d/neurodebian.sources.list \
  && export GNUPGHOME="$(mktemp -d)" \
  && echo "disable-ipv6" >> ${GNUPGHOME}/dirmngr.conf \
  && (apt-key adv --homedir $GNUPGHOME --recv-keys --keyserver hkp://pgpkeys.eu:80 0xA5D32F012649A5A9 \
  && apt-get update \
  && apt-get install -y git-annex-standalone git \
  && rm -rf /tmp/*
  
RUN python3 -m pip install --no-cache-dir datalad datalad-osf PyYAML

RUN git config --global --add user.name "nobrainerzoo" \
  && git config --global --add user.email "nobrainerzoo"

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR "/work"
LABEL maintainer="Hoda Rajaei <rajaei.hoda@gmail.com>"

