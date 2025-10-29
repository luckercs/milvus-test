package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/howeyc/gopass"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
	"google.golang.org/grpc"
)

const (
	msgFmt = "==== %s ====\n"
)

var (
	milvusAddr    string
	milvusUser    string
	milvusPass    string
	milvusTimeout int64
)

func usage() {
	fmt.Fprintf(flag.CommandLine.Output(), "使用示例:\n"+
		"# 创建集合\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag create\n"+
		"# 插入数据\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag insert -insert_batch_size 1000 -insert_batch_num 1000\n"+
		"# 插入数据到分区\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag insert -insert_batch_size 1000 -insert_batch_num 1000 -insert_with_partition -insert_partition_max 100000\n"+
		"# 查询集合\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag count\n"+
		"# 检索向量\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag search -topk 10\n"+
		"# 删除集合\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -collection_name hello_milvus -tag delete\n"+
		"# 业务模拟\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -password <milvus_pass> -collection_name hello_milvus -tag server -server_port 18089 -server_cron_enable\n"+
		"# 压测服务\n./milvus_bench -server <milvus_server> -port <milvus_port> -user root -password <milvus_pass> -collection_name hello_milvus -tag server -server_port 18089\n\n",
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

	argOperate := flag.String("tag", "count", "use mode: create|insert|count|search|delete|deletepartition|server")
	argCollectionName := flag.String("collection_name", "hello_milvus", "collection_name")
	argInsertBatchSize := flag.Int("insert_batch_size", 1000, "insert batch size")
	argInsertBatchNum := flag.Int("insert_batch_num", 3, "insert batch num")
	argInsertWithPartition := flag.Bool("insert_with_partition", false, "insert to partitions")
	argInsertPartitionMax := flag.Int64("insert_partition_max", 1000000, "insert_partition_max")
	argStartPartitionName := flag.String("start_partition_name", "", "start_partition_name, eg: 20250813")

	argIndexName := flag.String("index", "HNSW", "index name: IVF_SQ8|HNSW|IVF_FLAT")
	argMetricType := flag.String("metric", "IP", "metric_type: IP|L2|COSINE")
	argDim := flag.Int64("dim", 512, "vec dim")
	argTopk := flag.Int("topk", 10, "topk")

	argServerPort := flag.String("server_port", "8089", "server_port")
	argServerRunCron := flag.Bool("server_cron_enable", false, "server run cron task")
	argServerRunCronTestInsertFs := flag.Int("server_cron_insert_fs", 1, "Insert once every fs seconds, only server_cron_enable")
	argServerRunCronTestSearchFs := flag.Int("server_cron_search_fs", 60, "search once every fs seconds, only server_cron_enable")
	argServerInsertBatchsizePerSecond := flag.Int("server_cron_insertnum", 218, "server_insert_batchsize_persecond, only server_cron_enable")
	argServerTopk := flag.Int("server_cron_topk", 10, "server_topk, only server_cron_enable")
	argServerKeepDays := flag.Int("server_cron_keepdays", 7, "server_keep_days, only server_cron_enable")

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

	if *argOperate == "" {
		log.Printf(msgFmt, "tag 参数不能为空")
		os.Exit(1)
	}

	if *argCollectionName == "" {
		log.Printf(msgFmt, "collection_name 参数不能为空")
		os.Exit(1)
	}
	collectionName := *argCollectionName

	log.Printf(msgFmt, "start connecting to Milvus: "+milvusAddr)

	rand.Seed(time.Now().UnixNano())

	if *argOperate == "delete" {
		deleteCollection(collectionName)
	}

	if *argOperate == "deletepartition" {
		deleteCollectionPartition(collectionName, *argServerKeepDays)
	}

	if *argOperate == "create" {
		createCollection(collectionName, *argDim, *argIndexName, *argMetricType)
	}

	if *argOperate == "insert" {
		if *argInsertWithPartition {
			insertCollectionPartitionForHistory(collectionName, *argDim, *argInsertBatchSize, *argInsertBatchNum, *argInsertPartitionMax, *argStartPartitionName)
		} else {
			insertCollectionForHistory(collectionName, *argDim, *argInsertBatchSize, *argInsertBatchNum)
		}
	}

	if *argOperate == "search" {
		searchCollection(collectionName, *argDim, *argIndexName, *argMetricType, *argTopk)
	}

	if *argOperate == "count" {
		countCollection(collectionName)
	}

	if *argOperate == "server" {
		createCollection(collectionName, *argDim, *argIndexName, *argMetricType)
		var wg sync.WaitGroup

		if *argServerRunCron {
			tickerInsert := time.NewTicker(time.Duration(*argServerRunCronTestInsertFs) * time.Second)
			tickerSearch := time.NewTicker(time.Duration(*argServerRunCronTestSearchFs) * time.Second)
			tickerDelete := time.NewTicker(24 * time.Hour)
			wg.Add(1)
			go func() {
				defer wg.Done()
				log.Printf(msgFmt, "cron test task running: insert "+strconv.Itoa(*argServerInsertBatchsizePerSecond)+" entities/"+strconv.Itoa(*argServerRunCronTestInsertFs)+"s...")
				log.Printf(msgFmt, "cron test task running: search Top"+strconv.Itoa(*argServerTopk)+"/"+strconv.Itoa(*argServerRunCronTestSearchFs)+"s...")
				log.Printf(msgFmt, "cron test task running: delete old partitions, keep "+strconv.Itoa(*argServerKeepDays)+" days/24h...")
				for {
					select {
					case <-tickerInsert.C:
						insertCollectionPartition(collectionName, *argDim, *argServerInsertBatchsizePerSecond, 1)
					case <-tickerSearch.C:
						searchCollection(collectionName, *argDim, *argIndexName, *argMetricType, *argServerTopk)
					case <-tickerDelete.C:
						deleteCollectionPartition(collectionName, *argServerKeepDays)
					}
				}
			}()
		}
		gin.SetMode(gin.ReleaseMode)
		router := gin.Default()

		//  ?topk=10&partitions=2025xxxx
		router.GET("/search", searchCollectionForBench(collectionName, *argDim, *argIndexName, *argMetricType))
		//  ?num=1000
		router.GET("/insert", insertCollectionForBench(collectionName, *argDim))
		log.Printf(msgFmt, "webserver task running with port: "+*argServerPort)

		router.Run(":" + *argServerPort)
		wg.Wait()
	}

	log.Printf(msgFmt, "all finished")
}

func getMilvusClient() (context.Context, *milvusclient.Client, error) {
	ctx, _ := context.WithCancel(context.Background())
	milvusConfig := &milvusclient.ClientConfig{
		Address:  milvusAddr,
		Username: milvusUser,
		Password: milvusPass,
		DBName:   "default",

		DialOptions: []grpc.DialOption{
			grpc.WithBlock(),
			grpc.WithTimeout(time.Duration(milvusTimeout * 1000 * 1000)),
		},
	}
	c, err := milvusclient.New(ctx, milvusConfig)
	if err != nil {
		log.Fatalf(msgFmt, "Failed to connect to milvus: "+milvusAddr+", err=", err.Error())
	}
	return ctx, c, err
}

// for tag delete
func deleteCollection(collectionName string) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	err := c.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName))
	if err != nil {
		log.Fatalf(msgFmt, "failed to drop collection "+collectionName+", "+err.Error())
	} else {
		log.Printf(msgFmt, "delete collection: "+collectionName)
	}
}

// for tag deletepartition
func deleteCollectionPartition(collectionName string, daysAgo int) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	oldPartitionTime := time.Now().AddDate(0, 0, -daysAgo)
	oldPartitionName := oldPartitionTime.Format("20060102")

	c.ReleasePartitions(ctx, milvusclient.NewReleasePartitionsOptions(collectionName, oldPartitionName))
	err := c.DropPartition(ctx, milvusclient.NewDropPartitionOption(collectionName, oldPartitionName))
	if err != nil {
		log.Fatalf(msgFmt, "failed to drop partition "+collectionName+"["+oldPartitionName+"]"+", "+err.Error())
	} else {
		log.Printf(msgFmt, "delete partition: "+collectionName+"["+oldPartitionName+"]")
	}
}

// for tag create
func createCollection(collectionName string, dim int64, indexName string, metricName string) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)

	hasCollection, _ := c.HasCollection(ctx, milvusclient.NewHasCollectionOption(collectionName))
	if hasCollection {
		return
	}

	log.Printf(msgFmt, fmt.Sprintf("create collection, `%s`", collectionName))
	schema := entity.NewSchema().WithName(collectionName).WithDescription("milvus_bench").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(false)).
		WithField(entity.NewField().WithName("value").WithDataType(entity.FieldTypeFloatVector).WithDim(dim)).
		WithField(entity.NewField().WithName("deviceCode").WithDataType(entity.FieldTypeVarChar).WithMaxLength(50)).
		WithField(entity.NewField().WithName("createTime").WithDataType(entity.FieldTypeInt64))

	if err := c.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).WithConsistencyLevel(entity.DefaultConsistencyLevel)); err != nil {
		log.Fatalf("create collection failed, err: %v", err)
	}

	if indexName == "HNSW" {
		log.Printf(msgFmt, "start creating index HNSW")
		if _, err := c.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collectionName, "value", index.NewHNSWIndex(entity.MetricType(metricName), 64, 250))); err != nil {
			log.Fatalf("failed to create index, err: ", err)
		}
	} else if indexName == "IVF_SQ8" {
		log.Printf(msgFmt, "start creating index IVF_SQ8")
		if _, err := c.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collectionName, "value", index.NewIvfSQ8Index(entity.MetricType(metricName), 4096))); err != nil {
			log.Fatalf("failed to create index, err: ", err)
		}

	} else if indexName == "IVF_FLAT" {
		log.Printf(msgFmt, "start creating index IVF_FLAT")
		if _, err := c.CreateIndex(ctx, milvusclient.NewCreateIndexOption(collectionName, "value", index.NewIvfFlatIndex(entity.MetricType(metricName), 4096))); err != nil {
			log.Fatalf("failed to create index, err: ", err)
		}
	}
	log.Printf(msgFmt, "start loading collection")

	if _, err := c.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName).WithReplica(1)); err != nil {
		log.Fatalf("failed to load collection, err: ", err)
	}
}

// for tag insert
func insertCollectionPartitionForHistory(collectionName string, dim int64, insertBatchSize int, insertBatchNum int, insertPartitionMax int64, startPartition string) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	log.Printf(msgFmt, "start inserting random entities")
	var currentPartitionInsertTotal int64 = 0
	currentPartitionTime := time.Now().AddDate(0, 0, -1)
	currentPartitionName := currentPartitionTime.Format("20060102")
	if startTime, err := ParseDateString(startPartition); err == nil {
		log.Printf(msgFmt, "start partition: "+startPartition)
		currentPartitionTime = startTime
		currentPartitionName = currentPartitionTime.Format("20060102")
	}

	hasPartition, err := c.HasPartition(ctx, milvusclient.NewHasPartitionOption(collectionName, currentPartitionName))
	if err != nil {
		log.Fatalf("failed to check partition "+currentPartitionName+", err: %v", err)
	}
	if !hasPartition {
		err = c.CreatePartition(ctx, milvusclient.NewCreatePartitionOption(collectionName, currentPartitionName))
		if err != nil {
			log.Fatalf("failed to create partition "+currentPartitionName+", err: %v", err)
		}
	} else {
		queryResult, err := c.Query(ctx, milvusclient.NewQueryOption(collectionName).WithPartitions(currentPartitionName).WithOutputFields("count(*)"))
		if err != nil {
			log.Fatalf("failed to query partition count: "+currentPartitionName+", err: %v", err)
		}
		currentPartitionInsertTotal, _ = queryResult.GetColumn("count(*)").GetAsInt64(0)
	}
	for batchnum := 0; batchnum < insertBatchNum; batchnum++ {
		idList := make([]int64, 0, insertBatchSize)
		valueList := make([][]float32, 0, insertBatchSize)
		deviceCodeList := make([]string, 0, insertBatchSize)
		createTimeList := make([]int64, 0, insertBatchSize)
		// generate data
		for i := 0; i < insertBatchSize; i++ {
			idList = append(idList, int64(i)+int64(insertBatchSize)*int64(batchnum))
		}
		for i := 0; i < insertBatchSize; i++ {
			vec := make([]float32, 0, dim)
			for j := 0; j < int(dim); j++ {
				vec = append(vec, float32(rand.Float64()))
			}
			valueList = append(valueList, vec)
		}
		for i := 0; i < insertBatchSize; i++ { // 100-199
			deviceCodeList = append(deviceCodeList, "device"+strconv.Itoa(rand.Intn(100)+100))
		}
		for i := 0; i < insertBatchSize; i++ { // 毫秒时间戳
			createTimeList = append(createTimeList, time.Now().UnixMilli())
		}

		if _, err = c.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).WithPartition(currentPartitionName).
			WithInt64Column("id", idList).
			WithFloatVectorColumn("value", int(dim), valueList).
			WithVarcharColumn("deviceCode", deviceCodeList).
			WithInt64Column("createTime", createTimeList)); err != nil {
			log.Fatalf("failed to insert random data into "+collectionName+"["+currentPartitionName+"]"+", err: %v", err)
		}
		currentPartitionInsertTotal = currentPartitionInsertTotal + int64(insertBatchSize)
		if currentPartitionInsertTotal >= insertPartitionMax {
			currentPartitionInsertTotal = 0
			currentPartitionTime = currentPartitionTime.AddDate(0, 0, -1)
			currentPartitionName = currentPartitionTime.Format("20060102")
			log.Println("current partition reach max, start insert to new partition: " + currentPartitionName)

			hasPartition, err = c.HasPartition(ctx, milvusclient.NewHasPartitionOption(collectionName, currentPartitionName))
			if err != nil {
				log.Fatalf("failed to check partition "+currentPartitionName+", err: %v", err)
			}
			if !hasPartition {
				err = c.CreatePartition(ctx, milvusclient.NewCreatePartitionOption(collectionName, currentPartitionName))
				if err != nil {
					log.Fatalf("failed to create partition "+currentPartitionName+", err: %v", err)
				}
			} else {
				queryResult, err := c.Query(ctx, milvusclient.NewQueryOption(collectionName).WithPartitions(currentPartitionName).WithOutputFields("count(*)"))
				if err != nil {
					log.Fatalf("failed to query partition count: "+currentPartitionName+", err: %v", err)
				}
				currentPartitionInsertTotal, _ = queryResult.GetColumn("count(*)").GetAsInt64(0)
			}
		}
		log.Printf(msgFmt, collectionName+"["+currentPartitionName+"]"+"insert batch done [batchsize="+strconv.Itoa(insertBatchSize)+"] "+strconv.Itoa(batchnum+1)+" / "+strconv.Itoa(insertBatchNum)+", partitionTotal="+strconv.Itoa(int(currentPartitionInsertTotal))+" / Max="+strconv.Itoa(int(insertPartitionMax)))
	}
}

// for tag insert
func insertCollectionForHistory(collectionName string, dim int64, insertBatchSize int, insertBatchNum int) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	log.Printf(msgFmt, "start inserting random entities to default partition")
	for batchnum := 0; batchnum < insertBatchNum; batchnum++ {
		idList := make([]int64, 0, insertBatchSize)
		valueList := make([][]float32, 0, insertBatchSize)
		deviceCodeList := make([]string, 0, insertBatchSize)
		createTimeList := make([]int64, 0, insertBatchSize)
		// generate data
		for i := 0; i < insertBatchSize; i++ {
			idList = append(idList, int64(i)+int64(insertBatchSize)*int64(batchnum))
		}
		for i := 0; i < insertBatchSize; i++ {
			vec := make([]float32, 0, dim)
			for j := 0; j < int(dim); j++ {
				vec = append(vec, float32(rand.Float64()))
			}
			valueList = append(valueList, vec)
		}
		for i := 0; i < insertBatchSize; i++ { // 100-199
			deviceCodeList = append(deviceCodeList, "device"+strconv.Itoa(rand.Intn(100)+100))
		}
		for i := 0; i < insertBatchSize; i++ { // 毫秒时间戳
			createTimeList = append(createTimeList, time.Now().UnixMilli())
		}

		if _, err := c.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).
			WithInt64Column("id", idList).
			WithFloatVectorColumn("value", int(dim), valueList).
			WithVarcharColumn("deviceCode", deviceCodeList).
			WithInt64Column("createTime", createTimeList)); err != nil {
			log.Fatalf("failed to insert random data into "+collectionName+", err: %v", err)
		}

		log.Printf(msgFmt, collectionName+" insert batch done [batchsize="+strconv.Itoa(insertBatchSize)+"] "+strconv.Itoa(batchnum+1)+" / "+strconv.Itoa(insertBatchNum))
	}
}

// for server cron insert
func insertCollectionPartition(collectionName string, dim int64, insertBatchSize int, insertBatchNum int) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	log.Printf(msgFmt, "start inserting random entities")
	currentPartitionTime := time.Now()
	currentPartitionName := currentPartitionTime.Format("20060102")
	for batchnum := 0; batchnum < insertBatchNum; batchnum++ {
		idList := make([]int64, 0, insertBatchSize)
		valueList := make([][]float32, 0, insertBatchSize)
		deviceCodeList := make([]string, 0, insertBatchSize)
		createTimeList := make([]int64, 0, insertBatchSize)
		// generate data
		for i := 0; i < insertBatchSize; i++ {
			idList = append(idList, int64(i)+int64(insertBatchSize)*int64(batchnum))
		}
		for i := 0; i < insertBatchSize; i++ {
			vec := make([]float32, 0, dim)
			for j := 0; j < int(dim); j++ {
				vec = append(vec, float32(rand.Float64()))
			}
			valueList = append(valueList, vec)
		}
		for i := 0; i < insertBatchSize; i++ { // 100-199
			deviceCodeList = append(deviceCodeList, "device"+strconv.Itoa(rand.Intn(100)+100))
		}
		for i := 0; i < insertBatchSize; i++ { // 毫秒时间戳
			createTimeList = append(createTimeList, time.Now().UnixMilli())
		}

		if _, err := c.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).WithPartition(currentPartitionName).
			WithInt64Column("id", idList).
			WithFloatVectorColumn("value", int(dim), valueList).
			WithVarcharColumn("deviceCode", deviceCodeList).
			WithInt64Column("createTime", createTimeList)); err != nil {

			hasPartition, _ := c.HasPartition(ctx, milvusclient.NewHasPartitionOption(collectionName, currentPartitionName))
			if !hasPartition {
				err = c.CreatePartition(ctx, milvusclient.NewCreatePartitionOption(collectionName, currentPartitionName))
				if err != nil {
					log.Fatalf("failed to create partition "+currentPartitionName+", err: %v", err)
				}
				log.Printf(msgFmt, "create partition: "+collectionName+"["+currentPartitionName+"]")
				if _, err := c.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).WithPartition(currentPartitionName).
					WithInt64Column("id", idList).
					WithFloatVectorColumn("value", int(dim), valueList).
					WithVarcharColumn("deviceCode", deviceCodeList).
					WithInt64Column("createTime", createTimeList)); err != nil {
					log.Fatalf("failed to insert random data into "+collectionName+"["+currentPartitionName+"]"+", err: %v", err)
				}
			} else {
				log.Fatalf("failed to insert random data into "+collectionName+"["+currentPartitionName+"]"+", err: %v", err)
			}
		}
		log.Printf(msgFmt, collectionName+"["+currentPartitionName+"]"+" insert batch done [batchsize="+strconv.Itoa(insertBatchSize)+"] "+strconv.Itoa(batchnum+1)+" / "+strconv.Itoa(insertBatchNum))
	}
}

// for tag search and server cron search
func searchCollection(collectionName string, dim int64, indexName string, metricName string, topk int) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	log.Printf(msgFmt, "start search based on vector similarity")

	vecList := make([][]float32, 0, 1)
	vec := make([]float32, 0, dim)
	for j := 0; j < int(dim); j++ {
		vec = append(vec, float32(rand.Float64()))
	}
	vecList = append(vecList, vec)
	log.Println("search_vec: ==================")
	for _, row := range vecList {
		fmt.Print("[")
		for i, value := range row {
			fmt.Print(value)
			if i != len(row)-1 {
				fmt.Print(", ")
			}
		}
		fmt.Println("]")
	}

	vec2search := []entity.Vector{
		entity.FloatVector(vecList[len(vecList)-1]),
	}
	if indexName == "HNSW" {
		log.Println("search_params: ef: 1024 ==================")
		begin := time.Now()
		sRet, err := c.Search(ctx, milvusclient.NewSearchOption(collectionName, topk, vec2search).
			WithANNSField("value").WithOutputFields("id", "deviceCode", "createTime").
			WithSearchParam("ef", strconv.Itoa(1024)).WithConsistencyLevel(entity.ClBounded))
		end := time.Now()
		if err != nil {
			log.Fatalf("failed to search collection, err: %v", err)
		}

		log.Println("results: len=" + strconv.Itoa(len(sRet[0].Scores)))
		fmt.Println("id\tscore\tdeviceCode\tcreateTime")
		for i := 0; i < len(sRet[0].Scores); i++ {
			for _, res := range sRet {
				value1, _ := res.GetColumn("id").GetAsInt64(i)
				fmt.Print(value1)
				fmt.Print("\t")
				fmt.Print(res.Scores[i])
				fmt.Print("\t")
				value2, _ := res.GetColumn("deviceCode").GetAsString(i)
				fmt.Print(value2)
				fmt.Print("\t")
				value3, _ := res.GetColumn("createTime").GetAsInt64(i)
				fmt.Print(value3)
				fmt.Println()
			}
		}
		log.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)
	} else if indexName == "IVF_SQ8" {
		log.Println("search_params: nprobe: 128 ==================")
		begin := time.Now()
		sRet, err := c.Search(ctx, milvusclient.NewSearchOption(collectionName, topk, vec2search).
			WithANNSField("value").WithOutputFields("id", "deviceCode", "createTime").
			WithSearchParam("nprobe", strconv.Itoa(128)).WithConsistencyLevel(entity.ClBounded))
		end := time.Now()
		if err != nil {
			log.Fatalf("failed to search collection, err: %v", err)
		}

		log.Println("results: len=" + strconv.Itoa(len(sRet[0].Scores)))
		fmt.Println("id\tscore\tdeviceCode\tcreateTime")

		for i := 0; i < len(sRet[0].Scores); i++ {
			for _, res := range sRet {
				value1, _ := res.GetColumn("id").GetAsInt64(i)
				fmt.Print(value1)
				fmt.Print("\t")
				fmt.Print(res.Scores[i])
				fmt.Print("\t")
				value2, _ := res.GetColumn("deviceCode").GetAsString(i)
				fmt.Print(value2)
				fmt.Print("\t")
				value3, _ := res.GetColumn("createTime").GetAsInt64(i)
				fmt.Print(value3)
				fmt.Println()
			}
		}
		log.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)
	} else if indexName == "IVF_FLAT" {
		log.Println("search_params: nprobe: 128 ==================")
		begin := time.Now()
		sRet, err := c.Search(ctx, milvusclient.NewSearchOption(collectionName, topk, vec2search).
			WithANNSField("value").WithOutputFields("id", "deviceCode", "createTime").
			WithSearchParam("nprobe", strconv.Itoa(128)).WithConsistencyLevel(entity.ClBounded))
		end := time.Now()
		if err != nil {
			log.Fatalf("failed to search collection, err: %v", err)
		}

		log.Println("results: len=" + strconv.Itoa(len(sRet[0].Scores)))
		fmt.Println("id\tscore\tdeviceCode\tcreateTime")

		for i := 0; i < len(sRet[0].Scores); i++ {
			for _, res := range sRet {
				value1, _ := res.GetColumn("id").GetAsInt64(i)
				fmt.Print(value1)
				fmt.Print("\t")
				fmt.Print(res.Scores[i])
				fmt.Print("\t")
				value2, _ := res.GetColumn("deviceCode").GetAsString(i)
				fmt.Print(value2)
				fmt.Print("\t")
				value3, _ := res.GetColumn("createTime").GetAsInt64(i)
				fmt.Print(value3)
				fmt.Println()
			}
		}
		log.Printf("\tsearch latency: %dms\n", end.Sub(begin)/time.Millisecond)
	}
}

// for server rest insert
func insertCollectionForBench(collectionName string, dim int64) gin.HandlerFunc {
	return func(gincontext *gin.Context) {
		ctx, c, _ := getMilvusClient()
		defer c.Close(ctx)
		insertNumStr := gincontext.DefaultQuery("num", "1000")
		insertNum, err := strconv.Atoi(insertNumStr)
		if err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error_insert": err.Error()})
			return
		}
		if insertNum > 50000 {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error_insert": "num should less than 50000"})
			return
		}
		idList := make([]int64, 0, insertNum)
		valueList := make([][]float32, 0, insertNum)
		deviceCodeList := make([]string, 0, insertNum)
		createTimeList := make([]int64, 0, insertNum)

		for i := 0; i < insertNum; i++ {
			idList = append(idList, int64(i))
		}
		for i := 0; i < insertNum; i++ {
			vec := make([]float32, 0, dim)
			for j := 0; j < int(dim); j++ {
				vec = append(vec, float32(rand.Float64()))
			}
			valueList = append(valueList, vec)
		}
		for i := 0; i < insertNum; i++ { // 100-199
			deviceCodeList = append(deviceCodeList, "device"+strconv.Itoa(rand.Intn(100)+100))
		}
		for i := 0; i < insertNum; i++ { // 毫秒时间戳
			createTimeList = append(createTimeList, time.Now().UnixMilli())
		}

		if _, err := c.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).
			WithInt64Column("id", idList).
			WithFloatVectorColumn("value", int(dim), valueList).
			WithVarcharColumn("deviceCode", deviceCodeList).
			WithInt64Column("createTime", createTimeList)); err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error_insert": err.Error()})
			return
		}
		gincontext.JSON(http.StatusOK, gin.H{"message": "success insert " + strconv.Itoa(insertNum)})
	}
}

// for server rest search
func searchCollectionForBench(collectionName string, dim int64, indexName string, metricName string) gin.HandlerFunc {
	return func(gincontext *gin.Context) {
		ctx, c, _ := getMilvusClient()
		defer c.Close(ctx)
		query_topk_str := gincontext.DefaultQuery("topk", "10")
		query_topk, err := strconv.Atoi(query_topk_str)
		if err != nil {
			gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		query_partitions_str := gincontext.DefaultQuery("partitions", "")

		vecList := make([][]float32, 0, 1)
		vec := make([]float32, 0, dim)
		for j := 0; j < int(dim); j++ {
			vec = append(vec, float32(rand.Float64()))
		}
		vecList = append(vecList, vec)

		vec2search := []entity.Vector{
			entity.FloatVector(vecList[len(vecList)-1]),
		}
		if indexName == "HNSW" {
			log.Println("ANN search(HNSW), topk=" + strconv.Itoa(query_topk) + ", partitions=" + query_partitions_str)
			_, err := c.Search(ctx, milvusclient.NewSearchOption(collectionName, query_topk, vec2search).WithPartitions(SplitByComma(query_partitions_str)...).
				WithANNSField("value").WithOutputFields("id", "deviceCode", "createTime").
				WithSearchParam("ef", strconv.Itoa(1024)).WithConsistencyLevel(entity.ClBounded))
			if err != nil {
				gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			} else {
				gincontext.JSON(http.StatusOK, gin.H{"message": "success Top " + strconv.Itoa(query_topk)})
			}
		} else if indexName == "IVF_SQ8" {
			log.Println("ANN search(IVF_SQ8), topk=" + strconv.Itoa(query_topk) + ", partitions=" + query_partitions_str)
			_, err := c.Search(ctx, milvusclient.NewSearchOption(collectionName, query_topk, vec2search).WithPartitions(SplitByComma(query_partitions_str)...).
				WithANNSField("value").WithOutputFields("id", "deviceCode", "createTime").
				WithSearchParam("nprobe", strconv.Itoa(128)).WithConsistencyLevel(entity.ClBounded))
			if err != nil {
				gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			} else {
				gincontext.JSON(http.StatusOK, gin.H{"message": "success Top " + strconv.Itoa(query_topk)})
			}
		} else if indexName == "IVF_FLAT" {
			log.Println("ANN search(IVF_FLAT), topk=" + strconv.Itoa(query_topk) + ", partitions=" + query_partitions_str)
			_, err := c.Search(ctx, milvusclient.NewSearchOption(collectionName, query_topk, vec2search).WithPartitions(SplitByComma(query_partitions_str)...).
				WithANNSField("value").WithOutputFields("id", "deviceCode", "createTime").
				WithSearchParam("nprobe", strconv.Itoa(128)).WithConsistencyLevel(entity.ClBounded))
			if err != nil {
				gincontext.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			} else {
				gincontext.JSON(http.StatusOK, gin.H{"message": "success Top " + strconv.Itoa(query_topk)})
			}
		}
	}
}

// for tag count
func countCollection(collectionName string) {
	ctx, c, _ := getMilvusClient()
	defer c.Close(ctx)
	log.Printf(msgFmt, "start query count")
	queryResult, _ := c.Query(ctx, milvusclient.NewQueryOption(collectionName).WithOutputFields("count(*)"))
	fmt.Println("count(*)")
	value, _ := queryResult.GetColumn("count(*)").GetAsInt64(0)
	fmt.Println(value)
}

func SplitByComma(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	return parts
}

func ParseDateString(dateStr string) (time.Time, error) {
	formats := []string{
		"2006-01-02",
		"2006/01/02",
		"02-01-2006",
		"02/01/2006",
		"20060102",
		"2006-01-02 15:04:05",
	}

	for _, format := range formats {
		t, err := time.Parse(format, dateStr)
		if err == nil {
			return t, nil
		}
	}

	return time.Time{}, errors.New("error date format, eg: 20060102")
}
