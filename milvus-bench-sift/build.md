## (1) build

```shell
go mod tidy

set GOOS=linux
set GOARCH=amd64
go build -ldflags="-s -w" -a -o milvus_bench_sift_x86_64 milvus_bench_sift.go

set GOOS=linux
set GOARCH=arm64
go build -ldflags="-s -w" -a -o milvus_bench_sift_aarch64 milvus_bench_sift.go

docker build --platform=linux/amd64 --build-arg ARCH=x86_64 -f milvus_bench.dockerfile -t milvus-bench-sift:amd64-250910 .
docker build --platform=linux/arm64 --build-arg ARCH=aarch64 -f milvus_bench.dockerfile -t milvus-bench-sift:arm64-250910 .
```

## (3) get started

```shell
(1) write sift data to milvus
docker run -it --name milvus_bench_sift --rm milvus-bench-sift:amd64-250910 python milvus_bench_sift.py --milvus_uri http://localhost:19530 --milvus_token root:Milvus 

(2) get sift test data
docker run -it --name milvus_bench_sift --rm -v $(pwd)/testdata:/app/testdata milvus-bench-sift:amd64-250910 python milvus_bench_sift.py --op readAndSave

(3) run milvus_sift_bench_server
chmod +x milvus_bench_sift*
mv milvus_bench_sift* milvus_bench_sift
# 检索服务
./milvus_bench_sift -server <milvus_server> -port <milvus_port> -user root -op server -server_port 8089
cat<<EOF> curl.sh
curl http://localhost:8089?topk=10
EOF

(4) benchmark
./go-stress-testing-linux-x86_64 -c 100 -n 10 -p curl.sh

(5) delete collection
./milvus_bench_sift -server <milvus_server> -port <milvus_port> -user root -op delete

```
