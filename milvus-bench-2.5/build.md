## (1) build

```shell
go mod tidy
set GOOS=linux
set GOARCH=amd64
go build -a -o milvus_bench_2_5_x86_64 milvus-bench-2.5.go

go mod tidy
set GOOS=linux
set GOARCH=arm64
go build -a -o milvus_bench_2_5_aarch64 milvus-bench-2.5.go
```

## (2) get started

```shell
chmod +x milvus_bench

# 创建表
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag create

# 写入分区
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag insert -insert_batch_size 1000 -insert_batch_num 1000

# 写入数据到分区，会从当前日期的分区开始写起，当超过分区最大数，会往前倒退一天创建新分区继续写入
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag insert -insert_batch_size 1000 -insert_batch_num 1000 -insert_with_partition -insert_partition_max 100000

# 查询表count数
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag count

# 检索
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag search -topk 100

# 删除表
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag delete

# 后台运行负载任务，每秒进行写入操作，每分钟进行检索操作，滚动保留7个分区
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -password <milvus_pass> -collection_name hello_milvus -tag server -server_port 18089 -server_cron_enable -server_cron_insertnum 217 -server_cron_topk 100 -server_cron_keepdays 7

# 后台只运行restful-api
./milvus_bench -server <milvus_server> -port <milvus_port> -user root -password <milvus_pass> -collection_name hello_milvus -tag server -server_port 18089
# restful-api 查询
# /search?topk=10&partitions=20250817
# /insert?num=1000

```

