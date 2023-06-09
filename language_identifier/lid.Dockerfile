# docker build -f lid.Dockerfile -t zanonymous/mlrun-lid:1.2.1-torch2.0.1-redis .

ARG PYTORCH_VERSION=2.0.1
ARG CUDA_VERSION=11.7
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

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
    apt-get install --no-install-recommends -y gcc g++ libsndfile1 ffmpeg wget git && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

ARG SPEECHBRAIN_VERSION=0.5.14
ARG MLRUN_VERSION=1.2.1

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir speechbrain==${SPEECHBRAIN_VERSION}

# mlrun necessities
RUN pip3 install --no-cache-dir mlrun==${MLRUN_VERSION} redis pyOpenSSL==19.0.0 kafka-python==2.0.2 

WORKDIR /workspace

RUN mkdir /models
ADD models/* /models
