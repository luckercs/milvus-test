package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/howeyc/gopass"
	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"gonum.org/v1/hdf5"
	"google.golang.org/grpc"
)

const (
	msgFmt = "==== %s ====\n"
)

func main() {
	argMilvusServer := flag.String("server", "localhost", "milvus_server_hostname")
	argMilvusPort := flag.String("port", "19530", "milvus_server_port")
	argMilvusUser := flag.String("user", "root", "milvus_user")
	argMilvusPass := flag.String("password", "", "milvus_user_password")
	argMilvusTimeout := flag.Int64("timeout", 10000, "milvus_connect_timeout_ms")
	argSSL := flag.Bool("ssl", false, "enable SSL/TLS")

	argMilvusCollectionName := flag.String("collection", "sift_bench", "collection_name")
	argHDF5FilePath := flag.String("hdf5", "", "hdf5 file path")
	argHDF5DsNameInsert := flag.String("hdf5_ds_insert", "train", "hdf5 dataset name for insert")
	argHDF5DsNameSearch := flag.String("hdf5_ds_search", "test", "hdf5 dataset name for search")
	argSearchRandom := flag.Bool("search_with_random_vec", false, "enable search_with_random_vec")
	argMetricType := flag.String("metric", "IP", "metric type: IP|L2|COSINE")
	argM := flag.Int("m", 8, "hnsw m")
	argEf := flag.Int("ef", 200, "hnsw efConstruction")
	argdim := flag.Int64("dim", 128, "SIFT dimension")
	argServerPort := flag.String("server_port", "8089", "server port")
	argOperate := flag.String("op", "create", "use mode: create|insert|createAndInsert|delete|server")

	flag.Parse()
	rand.Seed(time.Now().UnixNano())
	if *argHDF5FilePath == "" {
		log.Printf(msgFmt, "input hdf5 file path with param hdf5")
		os.Exit(1)
	}

	if *argMilvusPass == "" {
		log.Print("input Milvus password: ")
		pswd, err := gopass.GetPasswd()
		if err != nil {
			log.Printf(msgFmt, "read password error: "+err.Error())
			os.Exit(1)
		}
		*argMilvusPass = string(pswd)
	}
	milvusAddr := *argMilvusServer + `:` + *argMilvusPort
	milvusUser := *argMilvusUser
	milvusPass := *argMilvusPass
	milvusTimeout := *argMilvusTimeout

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
		log.Println(msgFmt, "connect to milvus failed: "+err.Error())
		os.Exit(1)
	}

	if *argOperate == "create" {
		createCollection(ctx, cli, *argMilvusCollectionName, *argdim, *argMetricType, *argM, *argEf)
	}
	if *argOperate == "insert" {
		veclist := getHdf5Data(*argHDF5FilePath, *argHDF5DsNameInsert)
		insertCollection(ctx, cli, *argMilvusCollectionName, *argdim, veclist)
	}
	if *argOperate == "createAndInsert" {
		createCollection(ctx, cli, *argMilvusCollectionName, *argdim, *argMetricType, *argM, *argEf)
		veclist := getHdf5Data(*argHDF5FilePath, *argHDF5DsNameInsert)
		insertCollection(ctx, cli, *argMilvusCollectionName, *argdim, veclist)
	}
	if *argOperate == "delete" {
		deleteAllCollection(ctx, cli, *argMilvusCollectionName)
	}
	if *argOperate == "server" {
		gin.SetMode(gin.ReleaseMode)
		router := gin.Default()
		//  ?topk=10
		if !*argSearchRandom {
			veclists := getHdf5Data(*argHDF5FilePath, *argHDF5DsNameSearch)
			router.GET("/search", searchCollection(ctx, cli, *argMilvusCollectionName, *argdim, *argSearchRandom, veclists))
		} else {
			router.GET("/search", searchCollection(ctx, cli, *argMilvusCollectionName, *argdim, *argSearchRandom, [][]float32{}))
		}
		router.Run(":" + *argServerPort)
	}
}

func getHdf5Data(hdf5FilePath string, dsName string) [][]float32 {
	version, err := hdf5.LibVersion()
	if err != nil {
		panic(err)
	}
	log.Println("hdf5 version: " + version.String())

	if !hdf5.IsHDF5(hdf5FilePath) {
		panic("not hdf5 file")
	}

	f, err := hdf5.OpenFile(hdf5FilePath, hdf5.F_ACC_RDONLY)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	log.Println("hdf5 file opened: " + f.FileName())

	ds, err := f.OpenDataset(dsName)
	if err != nil {
		panic(err)
	}
	defer ds.Close()
	log.Println("hdf5 dataset opened: " + ds.Name())

	dimInfo, _, err := ds.Space().SimpleExtentDims()
	if err != nil {
		panic(err)
	}
	num := int(dimInfo[0])
	dim := int(dimInfo[1])
	log.Println("hdf5 dataset num:" + fmt.Sprint(num) + " dim:" + fmt.Sprint(dim))

	data := make([]float32, num*dim)
	ds.Read(&data[0])
	err = ds.Read(&data)
	if err != nil {
		panic(err)
	}

	data2d := make([][]float32, num)
	for i := 0; i < num; i++ {
		start := i * dim
		end := start + dim
		data2d[i] = data[start:end]
	}
	return data2d
}

func createCollection(ctx context.Context, cli *milvusclient.Client, collection_name string, dim int64, metricType string, m int, ef int) {
	res, err := cli.HasCollection(ctx, milvusclient.NewHasCollectionOption(collection_name))
	if err != nil {
		log.Printf(msgFmt, "has collection failed: "+err.Error())
		os.Exit(1)
	}
	if !res {
		schema := entity.NewSchema().WithName(collection_name).WithDescription(collection_name).
			WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
			WithField(entity.NewField().WithName("vec").WithDataType(entity.FieldTypeFloatVector).WithDim(dim))

		err := cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collection_name, schema))
		if err != nil {
			log.Printf(msgFmt, "create collection "+collection_name+" failed: "+err.Error())
			os.Exit(1)
		} else {
			log.Printf(msgFmt, "create collection "+collection_name+" success")
		}

		indexTask, err := cli.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collection_name, "vec", index.NewHNSWIndex(index.MetricType(metricType), m, ef)))
		if err != nil {
			log.Printf(msgFmt, "createIndex collection "+collection_name+"  failed: "+err.Error())
			os.Exit(1)
		}
		err = indexTask.Await(ctx)
		if err != nil {
			log.Printf(msgFmt, "createIndex collection "+collection_name+" await failed: "+err.Error())
			os.Exit(1)
		} else {
			log.Printf(msgFmt, "collection "+collection_name+" createIndex success with HSNW (m="+fmt.Sprint(m)+", efConstruction="+fmt.Sprint(ef)+"), metricType="+metricType)
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
		} else {
			log.Printf(msgFmt, "load collection "+collection_name+" success")
		}
	} else {
		log.Printf(msgFmt, "collection "+collection_name+" already exists")
	}
}

func insertCollection(ctx context.Context, cli *milvusclient.Client, collection_name string, dim int64, vecList [][]float32) {
	insertOptions := milvusclient.NewColumnBasedInsertOption(collection_name,
		column.NewColumnFloatVector("vec", int(dim), vecList),
	)
	res, err := cli.Insert(ctx, insertOptions)
	if err != nil {
		log.Printf("failed to insert data into " + collection_name + ", err: " + err.Error())
		os.Exit(1)
	} else {
		log.Println("insert data into " + collection_name + " success, insertCount=" + strconv.FormatInt(res.InsertCount, 10))
	}
}

func searchCollection(ctx context.Context, cli *milvusclient.Client, collection_name string, dim int64, withRandomSearch bool, queryVectors [][]float32) gin.HandlerFunc {
	return func(gincontext *gin.Context) {
		query_topk_str := gincontext.DefaultQuery("topk", "10")
		query_topk, err := strconv.Atoi(query_topk_str)
		if err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		var queryVector []float32
		if withRandomSearch {
			queryVector := make([]float32, dim)
			for j := 0; j < int(dim); j++ {
				queryVector[j] = float32(rand.Float64())
			}
		} else {
			randIndex := rand.Intn(len(queryVectors))
			queryVector = queryVectors[randIndex]
		}
		log.Println("queryVector=" + fmt.Sprint(queryVector))

		searchOption := milvusclient.NewSearchOption(collection_name, query_topk, []entity.Vector{entity.FloatVector(queryVector)}).
			WithANNSField("vec").
			WithOutputFields("id").WithConsistencyLevel(entity.ClBounded)
		res, err := cli.Search(ctx, searchOption)
		if err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		} else {
			log.Println("search ann IDs: ", res[0].IDs.FieldData().GetScalars())
			log.Println("search ann Scores: ", res[0].Scores)
			gincontext.JSON(http.StatusOK, gin.H{"message": "success Top " + strconv.Itoa(query_topk)})
		}
	}

}

func deleteAllCollection(ctx context.Context, cli *milvusclient.Client, collection_name string) {
	err := cli.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection_name))
	if err != nil {
		log.Printf(msgFmt, "drop collection "+collection_name+" failed: "+err.Error())
		os.Exit(1)
	} else {
		log.Printf(msgFmt, "drop collection "+collection_name+" success")
	}
}
