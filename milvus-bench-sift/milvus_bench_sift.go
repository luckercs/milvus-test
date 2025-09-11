package main

import (
	"context"
	"encoding/json"
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
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
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

	argMilvusCollectionName := flag.String("collection", "sift_bench", "collection_name")
	argTestDatasetFilePath := flag.String("hdf5_test_dataset_path", "./testdata/dataset.json", "search with hdf5 test dataset file path")
	argSearchRandom := flag.Bool("search_with_random_vec", false, "enable search_with_random_vec")
	argEfSearch := flag.Int("ef_search", 64, "hnsw efConstruction search param")
	argdim := flag.Int64("dim", 128, "SIFT dimension")
	argServerPort := flag.String("server_port", "8089", "server port")
	argOperate := flag.String("op", "server", "use mode: server|delete")

	flag.Parse()
	rand.Seed(time.Now().UnixNano())

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

	if *argOperate == "delete" {
		deleteAllCollection(milvusAddr, milvusUser, milvusPass, milvusTimeout, *argMilvusCollectionName)
	}
	if *argOperate == "server" {
		gin.SetMode(gin.ReleaseMode)
		router := gin.Default()
		//  ?topk=10
		if !*argSearchRandom {
			veclists, err := getHdf5TestData(*argTestDatasetFilePath)
			if err != nil {
				log.Printf(msgFmt, "getHdf5TestData failed: "+err.Error())
				os.Exit(1)
			}
			router.GET("/search", searchCollection(milvusAddr, milvusUser, milvusPass, milvusTimeout, *argMilvusCollectionName, *argdim, *argEfSearch, *argSearchRandom, veclists))
		} else {
			router.GET("/search", searchCollection(milvusAddr, milvusUser, milvusPass, milvusTimeout, *argMilvusCollectionName, *argdim, *argEfSearch, *argSearchRandom, nil))
		}
		log.Println("server start running, port=" + *argServerPort)
		router.Run(":" + *argServerPort)
	}
}

func searchCollection(milvusAddr string, milvusUser string, milvusPass string, milvusTimeout int64, collection_name string, dim int64, searchEf int, withRandomSearch bool, queryVectors []VecItem) gin.HandlerFunc {
	return func(gincontext *gin.Context) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var milvusConfig *milvusclient.ClientConfig
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

		cli, err := milvusclient.New(ctx, milvusConfig)
		defer cli.Close(ctx)
		if err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error": "connect to milvus failed" + err.Error()})
			return
		}

		query_topk_str := gincontext.DefaultQuery("topk", "10")
		query_topk, err := strconv.Atoi(query_topk_str)
		if err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		var queryVector []float32
		if withRandomSearch {
			queryVector = make([]float32, dim)
			for j := 0; j < int(dim); j++ {
				queryVector[j] = float32(rand.Float64())
			}
		} else {
			randIndex := rand.Intn(len(queryVectors))
			queryVector = queryVectors[randIndex].Vec
		}
		log.Println("queryVector=" + fmt.Sprint(queryVector))

		searchOption := milvusclient.NewSearchOption(collection_name, query_topk, []entity.Vector{entity.FloatVector(queryVector)}).
			WithANNSField("vec").
			WithOutputFields("id").WithConsistencyLevel(entity.ClBounded).
			WithSearchParam("ef", strconv.Itoa(searchEf))
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

func deleteAllCollection(milvusAddr string, milvusUser string, milvusPass string, milvusTimeout int64, collection_name string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var milvusConfig *milvusclient.ClientConfig
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

	cli, err := milvusclient.New(ctx, milvusConfig)
	defer cli.Close(ctx)
	if err != nil {
		log.Println(msgFmt, "connect to milvus failed: "+err.Error())
		os.Exit(1)
	}

	err = cli.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection_name))
	if err != nil {
		log.Printf(msgFmt, "drop collection "+collection_name+" failed: "+err.Error())
		os.Exit(1)
	} else {
		log.Printf(msgFmt, "drop collection "+collection_name+" success")
	}
}

type VecItem struct {
	Vec []float32 `json:"vec"`
}

func getHdf5TestData(jsonpath string) ([]VecItem, error) {
	data, err := os.ReadFile(jsonpath)
	if err != nil {
		fmt.Printf("read jsonpath error: %v\n", err)
		return nil, err
	}
	var items []VecItem
	err = json.Unmarshal(data, &items)
	if err != nil {
		fmt.Printf("JSON parse error: %v\n", err)
		return nil, err
	}
	fmt.Println("get hdf5 test data success, and size=" + fmt.Sprint(len(items)))
	return items, nil
}
