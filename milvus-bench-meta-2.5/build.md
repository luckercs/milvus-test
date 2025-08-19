## (1) build

```shell
set GOOS=linux
set GOARCH=amd64
go build -ldflags="-s -w" -a -o milvus_meta_bench_x86_64 milvus-meta-bench.go

set GOOS=linux
set GOARCH=arm64
go build -ldflags="-s -w" -a -o milvus_meta_bench_aarch64 milvus-meta-bench.go
```

## (2) get started

```shell
chmod +x milvus_meta_bench*
mv milvus_meta_bench* milvus_meta_bench

# 创建100张集合
./milvus_meta_bench -server <milvus_server> -port <milvus_port> -user root -colnum 100

```

