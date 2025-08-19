package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/howeyc/gopass"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"google.golang.org/grpc"

	"github.com/drhodes/golorem"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	msgFmt = "==== %s ====\n"
	dim    = 768
)

var (
	milvusAddr    string
	milvusUser    string
	milvusPass    string
	milvusTimeout int64
)

func usage() {
	fmt.Fprintf(flag.CommandLine.Output(), "使用示例:\n"+
		"# 创建集合\n./milvusworkload -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -operate create\n\n",
	)
	flag.PrintDefaults()
}

func main() {
	flag.Usage = usage
	argMilvusServer := flag.String("server", "localhost", "milvus_server_hostname")
	argMilvusPort := flag.String("port", "19530", "milvus_server_port")
	argMilvusUser := flag.String("user", "root", "milvus_user")
	argMilvusPass := flag.String("password", "", "milvus_user_password")
	argMilvusTimeout := flag.Int64("timeout", 10000, "milvus_connect_timeout_ms")
	argOperate := flag.String("op", "create", "use mode: create|delete|insert|createAndInsert")
	argInsertNum := flag.Int("num", 1000, "milvus insert number")
	argSSL := flag.Bool("ssl", false, "enable SSL/TLS")

	argCollectionNum_start := flag.Int("colnum_start", 1, "milvus_collection_num_start")
	argCollectionNum_end := flag.Int("colnum_end", 100, "milvus_collection_num_end")

	flag.Parse()
	if *argMilvusPass == "" {
		log.Print("请输入Milvus服务密码: ")
		pswd, err := gopass.GetPasswd()
		if err != nil {
			log.Printf(msgFmt, "读取密码时发生错误: "+err.Error())
			os.Exit(1)
		}
		*argMilvusPass = string(pswd)
	}
	milvusAddr = *argMilvusServer + `:` + *argMilvusPort
	milvusUser = *argMilvusUser
	milvusPass = *argMilvusPass
	milvusTimeout = *argMilvusTimeout
	cstart := *argCollectionNum_start
	cend := *argCollectionNum_end

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var milvusConfig *milvusclient.ClientConfig
	if *argSSL {
		milvusConfig = &milvusclient.ClientConfig{
			Address:       milvusAddr,
			Username:      milvusUser,
			Password:      milvusPass,
			DBName:        "default",
			EnableTLSAuth: true,

			DialOptions: []grpc.DialOption{
				grpc.WithBlock(),
				grpc.WithTimeout(time.Duration(milvusTimeout * 1000 * 1000)),
			},
		}
	} else {
		milvusConfig = &milvusclient.ClientConfig{
			Address:  milvusAddr,
			Username: milvusUser,
			Password: milvusPass,
			DBName:   "default",

			DialOptions: []grpc.DialOption{
				grpc.WithBlock(),
				grpc.WithTimeout(time.Duration(milvusTimeout * 1000 * 1000)),
			},
		}
	}

	cli, err := milvusclient.New(ctx, milvusConfig)
	defer cli.Close(ctx)
	if err != nil {
		log.Printf(msgFmt, "connect to milvus failed: "+err.Error())
		os.Exit(1)
	}

	if *argOperate == "create" {
		for i := cstart; i <= cend; i++ {
			collection_name := "hello_milvus_" + fmt.Sprint(i)
			createCollection(ctx, cli, collection_name)
			log.Printf(msgFmt, "create collection "+collection_name+" succeed, progress: "+fmt.Sprint(i)+"/"+fmt.Sprint(cend-cstart+1))
		}
	} else if *argOperate == "insert" {
		for i := cstart; i <= cend; i++ {
			collection_name := "hello_milvus_" + fmt.Sprint(i)
			insertCollection(ctx, cli, collection_name, *argInsertNum)
			log.Printf(msgFmt, "insert "+fmt.Sprint(*argInsertNum)+" data into "+collection_name+" succeed, progress: "+fmt.Sprint(i)+"/"+fmt.Sprint(cend-cstart+1))
		}
	} else if *argOperate == "createAndInsert" {
		for i := cstart; i <= cend; i++ {
			collection_name := "hello_milvus_" + fmt.Sprint(i)
			createCollection(ctx, cli, collection_name)
			insertCollection(ctx, cli, collection_name, *argInsertNum)
			log.Printf(msgFmt, "create and insert "+fmt.Sprint(*argInsertNum)+" data into "+collection_name+" succeed, progress: "+fmt.Sprint(i)+"/"+fmt.Sprint(cend-cstart+1))
		}
	} else if *argOperate == "delete" {
		deleteAllCollection(ctx, cli)
	}
	log.Printf(msgFmt, "all finished")
}

func createCollection(ctx context.Context, cli *milvusclient.Client, collection_name string) {
	res, err := cli.HasCollection(ctx, milvusclient.NewHasCollectionOption(collection_name))
	if err != nil {
		log.Printf(msgFmt, "has collection failed: "+err.Error())
		os.Exit(1)
	}
	if !res {
		schema := entity.NewSchema().WithName(collection_name).WithDescription(collection_name).
			WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
			WithField(entity.NewField().WithName("value").WithDataType(entity.FieldTypeFloatVector).WithDim(int64(dim))).
			WithField(entity.NewField().WithName("sparse").WithDataType(entity.FieldTypeSparseVector)).
			WithField(entity.NewField().WithName("indexText").WithDataType(entity.FieldTypeVarChar).WithMaxLength(8192 * 3).WithEnableMatch(true).WithEnableAnalyzer(true)).
			WithFunction(entity.NewFunction().WithName("bm25").WithInputFields("indexText").WithOutputFields("sparse").WithType(entity.FunctionTypeBM25)).
			WithField(entity.NewField().WithName("docId").WithDataType(entity.FieldTypeInt64))

		err := cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collection_name, schema))
		if err != nil {
			log.Printf(msgFmt, "create collection "+collection_name+" failed: "+err.Error())
			os.Exit(1)
		}

		indexTask, err := cli.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collection_name, "value", index.NewHNSWIndex("IP", 64, 250)))
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+"  failed: "+err.Error())
			os.Exit(1)
		}
		err = indexTask.Await(ctx)
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+" await failed: "+err.Error())
			os.Exit(1)
		}

		indexTask, err = cli.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collection_name, "sparse", index.NewSparseInvertedIndex("BM25", 0.2)))
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+"  failed: "+err.Error())
			os.Exit(1)
		}
		err = indexTask.Await(ctx)
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+" await failed: "+err.Error())
			os.Exit(1)
		}

		indexTask, err = cli.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collection_name, "indexText", index.NewTrieIndex()))
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+"  failed: "+err.Error())
			os.Exit(1)
		}
		err = indexTask.Await(ctx)
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+" await failed: "+err.Error())
			os.Exit(1)
		}

		indexTask, err = cli.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collection_name, "docId", index.NewSortedIndex()))
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+"  failed: "+err.Error())
			os.Exit(1)
		}
		err = indexTask.Await(ctx)
		if err != nil {
			log.Printf(msgFmt, "CreateIndex collection "+collection_name+" await failed: "+err.Error())
			os.Exit(1)
		}

		loadTask, err := cli.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collection_name))
		if err != nil {
			log.Printf(msgFmt, "load collection "+collection_name+" failed: "+err.Error())
			os.Exit(1)
		}
		err = loadTask.Await(ctx)
		if err != nil {
			log.Printf(msgFmt, "load collection "+collection_name+" await failed: "+err.Error())
			os.Exit(1)
		}
	}
}

func insertCollection(ctx context.Context, cli *milvusclient.Client, collection_name string, num int) {
	idList := make([]int64, 0, num)
	valueList := make([][]float32, 0, num)
	indexTextList := make([]string, 0, num)
	docIdList := make([]int64, 0, num)

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < num; i++ {
		idList = append(idList, int64(i))
		docIdList = append(docIdList, rand.Int63())
		indexTextList = append(indexTextList, lorem.Sentence(10, 20))
	}
	for i := 0; i < num; i++ {
		vec := make([]float32, 0, dim)
		for j := 0; j < dim; j++ {
			vec = append(vec, rand.Float32())
		}
		valueList = append(valueList, vec)
	}

	insertOptions := milvusclient.NewColumnBasedInsertOption(collection_name,
		column.NewColumnInt64("id", idList),
		column.NewColumnFloatVector("value", dim, valueList),
		column.NewColumnVarChar("indexText", indexTextList),
		column.NewColumnInt64("docId", docIdList),
	)

	_, err := cli.Insert(ctx, insertOptions)
	if err != nil {
		log.Printf("failed to insert data into " + collection_name + ", err: " + err.Error())
		os.Exit(1)
	}
}

func deleteAllCollection(ctx context.Context, cli *milvusclient.Client) {
	collectionNames, err := cli.ListCollections(ctx, milvusclient.NewListCollectionOption())
	if err != nil {
		log.Printf(msgFmt, "list collections failed: "+err.Error())
		os.Exit(1)
	}
	for _, collectionName := range collectionNames {
		err = cli.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName))
		if err != nil {
			log.Printf(msgFmt, "drop collections "+collectionName+" failed: "+err.Error())
			os.Exit(1)
		}
	}
}
