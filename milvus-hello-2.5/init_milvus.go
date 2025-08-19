package main

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"time"

	"github.com/howeyc/gopass"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	__msgFmt = "==== %s ====\n"
)

func main() {
	arg1 := flag.String("server", "localhost", "milvus_server_hostname")
	arg2 := flag.String("port", "19530", "milvus_server_port")
	arg3 := flag.String("passwordold", "Milvus", "milvus_password_old")
	arg4 := flag.String("passwordnew", "", "milvus_password_new")
	arg5 := flag.Int64("timeout", 10000, "milvus_connect_timeout_ms")
	arg6 := flag.String("user", "root", "milvus_user")

	argSSL := flag.Bool("ssl", false, "enable SSL/TLS")
	certpath := flag.String("certpath", "", "TLS: root cert path, ca.pem")
	servername := flag.String("servername", "", "TLS: servername")
	skipverify := flag.Bool("skipverify", false, "Insecure Skip Verify")

	flag.Parse()
	milvusAddr := *arg1 + `:` + *arg2
	if *servername == "" {
		*servername = *arg1
	}

	if *arg4 == "" {
		log.Printf("Input milvus server new password: ")
		pswd, err := gopass.GetPasswd()
		if err != nil || len(string(pswd)) == 0 {
			log.Printf("read password error or password can not be null: " + err.Error())
			os.Exit(1)
		}
		*arg4 = string(pswd)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var milvusConfig *milvusclient.ClientConfig
	if *argSSL {
		log.Printf(__msgFmt, "start connecting to Milvus: https://"+milvusAddr)
		certPool := x509.NewCertPool()
		if !*skipverify {
			caCert, err := ioutil.ReadFile(*certpath)
			if err != nil {
				log.Printf("Failed to read CA certificate: %v", err)
				os.Exit(1)
			}
			if !certPool.AppendCertsFromPEM(caCert) {
				log.Printf("Failed to append CA certificate")
				os.Exit(1)
			}
		}

		tlsConfig := &tls.Config{
			RootCAs:            certPool,
			ServerName:         *servername,
			InsecureSkipVerify: *skipverify,
		}

		milvusConfig = &milvusclient.ClientConfig{
			Address:       milvusAddr,
			Username:      *arg6,
			Password:      *arg3,
			DBName:        "default",
			EnableTLSAuth: true,

			DialOptions: []grpc.DialOption{
				grpc.WithBlock(),
				grpc.WithTimeout(time.Duration(*arg5 * 1000 * 1000)),
				grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)),
			},
		}
	} else {
		log.Printf(__msgFmt, "start connecting to Milvus: http://"+milvusAddr)
		milvusConfig = &milvusclient.ClientConfig{
			Address:  milvusAddr,
			Username: *arg6,
			Password: *arg3,
			DBName:   "default",

			DialOptions: []grpc.DialOption{
				grpc.WithBlock(),
				grpc.WithTimeout(time.Duration(*arg5 * 1000 * 1000)),
			},
		}
	}

	c, err := milvusclient.New(ctx, milvusConfig)
	defer c.Close(ctx)
	if err != nil {
		log.Printf(__msgFmt, "connect to milvus failed: "+err.Error())
		os.Exit(1)
	}

	err = c.UpdatePassword(ctx, milvusclient.NewUpdatePasswordOption(*arg6, *arg3, *arg4))
	if err != nil {
		log.Printf(__msgFmt, "failed to init milvus password: "+err.Error())
		os.Exit(1)
	} else {
		log.Printf(__msgFmt, "init Milvus successful")
	}
}
