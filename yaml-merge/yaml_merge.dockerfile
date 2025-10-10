FROM bitnami/python:latest

ARG ARCH=x86_64
# ARG ARCH=arm64

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install ruamel.yaml
COPY ./yaml_merge.py /app/yaml_merge.py

WORKDIR /app
CMD [ "/bin/bash" ]

LABEL dev="2269732520@qq.com"

# docker build --platform=linux/amd64 --build-arg ARCH=x86_64 -f yaml_merge.dockerfile -t yaml_merge:1.0 .
# docker build --platform=linux/arm64 --build-arg ARCH=aarch64 -f yaml_merge.dockerfile -t yaml_merge:1.0 .

