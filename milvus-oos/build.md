## (1) build

```shell
go mod tidy
set GOOS=linux
set GOARCH=amd64
go build -ldflags="-s -w" -a -o hello_oos_x86_64 hello_oos.go

set GOOS=linux
set GOARCH=arm64
go build -ldflags="-s -w" -a -o hello_oos_aarch64 hello_oos.go

```
