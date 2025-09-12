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
docker run -it --name milvus_bench_sift --rm milvus-bench-sift:amd64-250910 python milvus_bench_sift.py --milvus_uri http://<MILVUS_SERVER>:<MILVUS_PORT>

(2) get sift test data
docker run -it --name milvus_bench_sift --rm -v $(pwd)/testdata:/app/testdata milvus-bench-sift:amd64-250910 python milvus_bench_sift.py --op readAndSave --hdf5_dataset test

(3) run test server
docker run -itd --name milvus_bench_sift --rm  milvus-bench-sift:amd64-250910 /bin/bash
docker cp milvus_bench_sift:/app/milvus_bench_sift ./
docker cp milvus_bench_sift:/app/go-stress-testing-linux ./
docker rm -f milvus_bench_sift

nohup ./milvus_bench_sift -server <MILVUS_SERVER> -port <MILVUS_PORT> -user root -password <MILVUS_PASSWORD>  -op server -server_port 8089 > log 2>&1 & 

cat<<EOF> curl.sh
curl http://localhost:8089/search?topk=10
EOF

(4) benchmark
./go-stress-testing-linux -c 1000 -n 10 -p curl.sh

(5) delete collection
./milvus_bench_sift -server <MILVUS_SERVER> -port <MILVUS_PORT> -user root -op delete

(6) stop test server
ps -ef | grep milvus_bench_sift | grep -v grep | awk '{print $2}' | xargs kill -9
```
