FROM bitnami/python:latest

ARG ARCH=x86_64
# ARG ARCH=arm64

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install pymilvus==2.5.14
RUN pip install h5py
RUN mkdir -p /app/testdata

# https://github.com/erikbern/ann-benchmarks/#data-sets
# download sift 128 dataset
# https://ann-benchmarks.com/sift-128-euclidean.hdf5
ADD ./sift-128-euclidean.hdf5 /app/sift-128-euclidean.hdf5
ADD ./milvus_bench_sift.py /app/milvus_bench_sift.py
# build first
ADD ./milvus_bench_sift_${ARCH} /app/milvus_bench_sift
# download from https://github.com/luckercs/go-stress-testing/releases
ADD ./go-stress-testing-linux-${ARCH} /app/go-stress-testing-linux
RUN chmod +x /app/milvus_bench_sift /app/go-stress-testing-linux

WORKDIR /app
CMD [ "/bin/bash" ]

LABEL dev="2269732520@qq.com"

# docker build --platform=linux/amd64 --build-arg ARCH=x86_64 -f milvus_bench.dockerfile -t milvus-bench-sift:amd64-250910 .
# docker build --platform=linux/arm64 --build-arg ARCH=aarch64 -f milvus_bench.dockerfile -t milvus-bench-sift:arm64-250910 .