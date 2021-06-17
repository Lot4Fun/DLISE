FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    lhasa \
    libgeos-3.6.2 \
    libgeos-dev \
    python3.8 \
    python3.8-dev \
    python3-pip \
 && apt-get -y clean \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN ln -f -n -s /usr/bin/python3.8 /usr/bin/python3

WORKDIR /DLISE
COPY ./ /DLISE/

RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade pip setuptools
RUN pip3 install "git+https://github.com/matplotlib/basemap.git"

RUN mkdir -p /archive/DLISE
