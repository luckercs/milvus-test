## (1) build

```shell
go mod tidy
set GOOS=linux
set GOARCH=amd64
go build -ldflags="-s -w" -a -o hello_milvus_x86_64 hello_milvus.go
go build -ldflags="-s -w" -a -o hello_milvus_gpu_x86_64 hello_milvus_gpu.go
go build -ldflags="-s -w" -a -o init_milvus_x86_64 init_milvus.go

set GOOS=linux
set GOARCH=arm64
go build -ldflags="-s -w" -a -o hello_milvus_aarch64 hello_milvus.go
go build -ldflags="-s -w" -a -o init_milvus_aarch64 init_milvus.go

```
