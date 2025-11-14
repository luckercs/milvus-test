package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/howeyc/gopass"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	msgFmt                     = "==== %s ====\n"
	idCol, f1Col, embeddingCol = "ID", "f1", "embeddings"
)

func main() {
	arg1 := flag.String("server", "localhost", "milvus_server_hostname")
	arg2 := flag.String("port", "19530", "milvus_server_port")
	arg3 := flag.String("user", "root", "milvus_user")
	arg4 := flag.String("password", "", "milvus_user_password")
	arg5 := flag.Int64("timeout", 10000, "milvus_connect_timeout_ms")
	arg6 := flag.Int("num", 300, "number of entities to insert")
	arg7 := flag.Int("dim", 4, "dimension of vectors")
	arg8 := flag.Int("topk", 3, "top k results to return")
	arg9 := flag.String("colname", "hello_milvus", "milvus collection name")
	saveCol := flag.Bool("save", false, "save collection, not delete collection")
	argSSL := flag.Bool("ssl", false, "enable SSL/TLS")
	certpath := flag.String("certpath", "", "TLS: root cert path, ca.pem")
	servername := flag.String("servername", "", "TLS: servername")
	skipverify := flag.Bool("skipverify", false, "Insecure Skip Verify")
	printvec := flag.Bool("showvector", false, "enable print embedding vectors")

	flag.Parse()
	milvusAddr := *arg1 + `:` + *arg2
	nEntities := *arg6
	dim := *arg7
	topK := *arg8
	collectionName := *arg9
	if *servername == "" {
		*servername = *arg1
	}

	if *arg4 == "" {
		log.Printf("Input milvus server password: ")
		pswd, err := gopass.GetPasswd()
		if err != nil {
			log.Printf("read password error: " + err.Error())
			os.Exit(1)
		}
		*arg4 = string(pswd)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var milvusConfig *milvusclient.ClientConfig
	if *argSSL {
		log.Printf(msgFmt, "start connecting to Milvus: https://"+milvusAddr)
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
			Username:      *arg3,
			Password:      *arg4,
			DBName:        "default",
			EnableTLSAuth: true,

			DialOptions: []grpc.DialOption{
				grpc.WithBlock(),
				grpc.WithTimeout(time.Duration(*arg5 * 1000 * 1000)),
				grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)),
			},
		}
	} else {
		log.Printf(msgFmt, "start connecting to Milvus: http://"+milvusAddr)
		milvusConfig = &milvusclient.ClientConfig{
			Address:  milvusAddr,
			Username: *arg3,
			Password: *arg4,
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
		log.Printf(msgFmt, "connect to milvus failed: "+err.Error())
		os.Exit(1)
	}

	has, err := c.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if err != nil {
		log.Printf("failed to check collection exists, err: %v", err)
		os.Exit(1)
	}
	if has {
		err = c.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName))
		if err != nil {
			log.Printf("failed to drop collection, err: %v", err)
			os.Exit(1)
		}
	}

	log.Printf(msgFmt, fmt.Sprintf("start create collection with hnsw index, `%s`", collectionName))
	schema := entity.NewSchema().WithName(collectionName).WithDescription(collectionName + "_demo").
		WithField(entity.NewField().WithName(idCol).WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
		WithField(entity.NewField().WithName(f1Col).WithDataType(entity.FieldTypeDouble)).
		WithField(entity.NewField().WithName(embeddingCol).WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dim)))

	indexOption1 := milvusclient.NewCreateIndexOption(collectionName, embeddingCol, index.NewHNSWIndex("IP", 64, 250))

	err = c.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).WithIndexOptions(indexOption1))
	if err != nil {
		log.Printf(msgFmt, "create collection with index "+collectionName+" failed: "+err.Error())
		os.Exit(1)
	}

	log.Printf(msgFmt, "start loading collection")
	_, err = c.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		log.Printf(msgFmt, "load collection "+collectionName+" failed: "+err.Error())
		os.Exit(1)
	}

	log.Printf(msgFmt, "start insert entities, num="+strconv.Itoa(nEntities)+", dim="+strconv.Itoa(dim))
	idList := make([]int64, 0, nEntities)
	f1List := make([]float64, 0, nEntities)
	embeddingList := make([][]float32, 0, nEntities)

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < nEntities; i++ {
		idList = append(idList, int64(i))
	}
	for i := 0; i < nEntities; i++ {
		f1List = append(f1List, rand.Float64())
	}
	for i := 0; i < nEntities; i++ {
		vec := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vec = append(vec, rand.Float32())
		}
		embeddingList = append(embeddingList, vec)
	}

	insertOptions := milvusclient.NewColumnBasedInsertOption(collectionName,
		column.NewColumnInt64(idCol, idList),
		column.NewColumnDouble(f1Col, f1List),
		column.NewColumnFloatVector(embeddingCol, dim, embeddingList),
	)

	if _, err := c.Insert(ctx, insertOptions); err != nil {
		log.Printf("failed to insert data into " + collectionName + ", err: " + err.Error())
		os.Exit(1)
	}

	log.Printf(msgFmt, "start ann search, topk="+strconv.Itoa(topK))
	vec2search := []entity.Vector{entity.FloatVector(embeddingList[len(embeddingList)-1])}
	log.Printf(msgFmt, "search vector: "+float32SliceToString(embeddingList[len(embeddingList)-1]))
	searchOption := milvusclient.NewSearchOption(collectionName, topK, vec2search).
		WithANNSField(embeddingCol).
		WithConsistencyLevel(entity.ClStrong).
		WithOutputFields(idCol, f1Col, embeddingCol)
	begin := time.Now()
	sRets, err := c.Search(ctx, searchOption)
	if err != nil {
		log.Printf("failed to search collection, err: %v", err)
		os.Exit(1)
	}
	end := time.Now()

	log.Println("results:==================================")
	if *printvec {
		fmt.Println(idCol + "\t" + f1Col + "\t" + embeddingCol + "\t" + "score")
	} else {
		fmt.Println(idCol + "\t" + f1Col + "\t" + "score")
	}
	for _, res := range sRets {
		if *printvec {
			allvectors := res.GetColumn(embeddingCol).FieldData().GetVectors().GetFloatVector().GetData()
			splitFloatSlice := splitFloatSlice(allvectors, dim)
			for i := range res.ResultCount {
				fmt.Print(res.GetColumn(idCol).FieldData().GetScalars().GetLongData().GetData()[i])
				fmt.Print("\t")
				fmt.Print(res.GetColumn(f1Col).FieldData().GetScalars().GetDoubleData().GetData()[i])
				fmt.Print("\t")
				fmt.Print(splitFloatSlice[i])
				fmt.Print("\t")
				fmt.Print(res.Scores[i])
				fmt.Println()
			}
		} else {
			for i := range res.ResultCount {
				fmt.Print(res.GetColumn(idCol).FieldData().GetScalars().GetLongData().GetData()[i])
				fmt.Print("\t")
				fmt.Print(res.GetColumn(f1Col).FieldData().GetScalars().GetDoubleData().GetData()[i])
				fmt.Print("\t")
				fmt.Print(res.Scores[i])
				fmt.Println()
			}
		}
	}
	log.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)

	if !*saveCol {
		log.Printf(msgFmt, "drop collection "+collectionName)
		if err = c.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName)); err != nil {
			log.Println("failed to drop collection "+collectionName+", err:", err)
			os.Exit(1)
		}
	}
	log.Printf("All Finished")
}

func splitFloatSlice(input []float32, dim int) [][]float32 {
	var result [][]float32
	for i := 0; i < len(input); i += dim {
		end := i + dim
		result = append(result, input[i:end])
	}
	return result
}

func float32SliceToString(slice []float32) string {
	if len(slice) == 0 {
		return "[]"
	}
	buf := bytes.NewBufferString("[")
	buf.WriteString(strconv.FormatFloat(float64(slice[0]), 'f', 7, 32))
	for _, num := range slice[1:] {
		buf.WriteString(",")
		buf.WriteString(strconv.FormatFloat(float64(num), 'f', 7, 32))
	}
	buf.WriteByte(']')
	return buf.String()
}
