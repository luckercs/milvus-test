package main

import (
	"context"
	"fmt"
	"log"

	"github.com/minio/minio-go/v7"
	minioCred "github.com/minio/minio-go/v7/pkg/credentials"
)

func main() {
	endpoint := "beijing-5.zos.ctyun.cn:443"
	accessKey := "XX"
	secretKey := "XXX"

	minioClient, err := minio.New(endpoint, &minio.Options{
		Creds:  minioCred.NewStaticV4(accessKey, secretKey, ""),
		Secure: true, // ← 自动使用 HTTPS (端口 443)

	})
	if err != nil {
		log.Fatalln("创建客户端失败:", err)
	}

	buckets, err := minioClient.ListBuckets(context.Background())
	if err != nil {
		log.Fatalln("列出 buckets 失败:", err)
	}

	for _, b := range buckets {
		fmt.Println("Bucket:", b.Name)
	}

}
