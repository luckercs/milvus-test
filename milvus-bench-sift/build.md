## (1) dataset
```shell
https://github.com/erikbern/ann-benchmarks/#data-sets
download sift 128 dataset
https://ann-benchmarks.com/sift-128-euclidean.hdf5
```
HDF5数据集名称,也可通过 HDFView 等工具查看文件内部的 Group/Dataset 层级
```shell
# pip install h5py
import h5py
import numpy as np
hdf5_path = "/sift-128-euclidean.hdf5"
f=h5py.File(hdf5_path, 'r')
f.keys()
f['distances']
f['distances'][:]
```

## (2) build

1> hdf5 env
```shell
https://www.hdfgroup.org/download-hdf5/
https://github.com/HDFGroup/hdf5/releases

# for ubuntu:
apt -y install libhdf5-dev
 yum install hdf5-devel
```

2> build

```shell
go mod tidy

set GOOS=linux
set GOARCH=amd64
go build -ldflags="-s -w" -a -o milvus_bench_sift_x86_64 milvus_bench_sift.go
```

## (3) get started

```shell
chmod +x milvus_bench_sift*
mv milvus_bench_sift* milvus_bench_sift

# 创建集合并写入数据
./milvus_bench_sift -server <milvus_server> -port <milvus_port> -user root -hdf5 <hdf5_file_path> -hdf5_ds_insert train -op createAndInsert

# 删除集合
./milvus_bench_sift -server <milvus_server> -port <milvus_port> -user root -op delete

# 检索服务
./milvus_bench_sift -server <milvus_server> -port <milvus_port> -user root -op server -server_port 8089 -search_with_random_vec 
curl http://localhost:8089?topk=10
```
