FROM bitnami/python:latest


RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install vectordb-bench==1.0.11
RUN mkdir -p /app/testdata

ENV DATASET_SOURCE=ALIYUNOSS
ENV DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset

# download dataset to local path
# /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/

# run test
# vectordbbench --help

WORKDIR /app
CMD [ "/bin/bash" ]

LABEL dev="2269732520@qq.com"

# docker build --platform=linux/amd64 -f vdbbench.dockerfile -t vdbbench:amd64-251030 .
# docker build --platform=linux/arm64 -f vdbbench.dockerfile -t vdbbench:arm64-251030 .