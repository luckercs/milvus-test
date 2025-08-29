FROM bitnami/python:latest

RUN apt clean && apt update
RUN apt -y install vim
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install h5py
RUN pip install protobuf==3.20.0
RUN pip install grpcio-tools==1.67.1
RUN python3 -m pip install pymilvus==2.5.14
