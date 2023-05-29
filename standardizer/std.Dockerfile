# docker build -f std.Dockerfile -t dleongsh/mlrun/std:1.2.1 .

ARG PYTORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

FROM python:3.10.11-slim-buster

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONFAULTHANDLER 1
ENV TZ=Asia/Singapore

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt -y update && \
    apt-get install --no-install-recommends -y gcc libsndfile1 libsox-fmt-all ffmpeg sox wget && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

ARG MLRUN_VERSION=1.2.1

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir sox && \
    pip3 install --no-cache-dir mlrun==${MLRUN_VERSION} 

WORKDIR /workspace
