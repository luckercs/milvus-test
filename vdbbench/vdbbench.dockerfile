FROM bitnami/python:latest

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install vectordb-bench==1.0.11
RUN sed -i '61,67 s/^/#/' /opt/bitnami/python/lib/python3.13/site-packages/vectordb_bench/backend/data_source.py
RUN apt update
RUN apt -y install vim

ENV DATASET_SOURCE=ALIYUNOSS
ENV DATASET_LOCAL_DIR=/tmp/vectordb_bench/dataset
ENV NUM_PER_BATCH=1000

# download dataset to local path
# /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/

# run test
# vectordbbench --help

WORKDIR /app
CMD [ "/bin/bash" ]

LABEL dev="2269732520@qq.com"

# docker build --platform=linux/amd64 -f vdbbench.dockerfile -t vdbbench:1.0.11 .
# docker build --platform=linux/arm64 -f vdbbench.dockerfile -t vdbbench:1.0.11 .