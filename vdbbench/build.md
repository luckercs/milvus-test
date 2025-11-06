https://github.com/zilliztech/VectorDBBench

（1）上传数据集并挂载到路径下
/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/

# 使用以下python代码离线下载数据集，需要知道每个数据集文件的名字
import oss2
bucket=oss2.Bucket(oss2.AnonymousAuth(), "assets.zilliz.com.cn/benchmark/", "benchmark", True)
bucket.get_object_to_file("benchmark/cohere_medium_1m/test.parquet", "/tmp/test.parquet")

# 或者使用wget链接下载, cohere1m数据集下载示例
wget https://assets.zilliz.com/benchmark/cohere_medium_1m/test.parquet --no-check-certificate
wget https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors.parquet --no-check-certificate
wget https://assets.zilliz.com/benchmark/cohere_medium_1m/shuffle_train.parquet --no-check-certificate
wget https://assets.zilliz.com/benchmark/cohere_medium_1m/scalar_labels.parquet --no-check-certificate
上述命令下载的数据集文件放在 /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/ 目录下。

# 其余数据集类似
LAION(768) GIST(960) Cohere(768) Bioasq(1024) Glove(200) SIFT(128) OpenAI(1536)
查看对应数据集参数： https://github.com/zilliztech/VectorDBBench/blob/main/vectordb_bench/backend/dataset.py
with_gt 则包含 neighbors.parquet
with_scalar_labels 则包含 scalar_labels.parquet
_size_label 对应数据集大小

（2）查看所有测试类型
vectordbbench --help
# 容量测试
[CapacityDim128|CapacityDim960|
# 普通检索
Performance768D100M|Performance768D10M|Performance768D1M|
Performance1536D500K|Performance1536D5M|
Performance1024D1M|Performance1024D10M|Performance1536D50K|
# 混合检索
Performance768D10M1P|Performance768D1M1P|Performance768D10M99P|Performance768D1M99P|
Performance1536D500K1P|Performance1536D5M1P|Performance1536D500K99P|Performance1536D5M99P|
# 其他测试
PerformanceCustomDataset|StreamingPerformanceCase|LabelFilterPerformanceCase|NewIntFilterPerformanceCase]


（3）运行测试
# 参考华为高斯库测试： https://docs.opengauss.org/zh/docs/latest/docs/DataVec/openGauss-VectorDBBench.html
vectordbbench milvushnsw --uri --user-name  --password --replica-number 1 --m --ef-construction   --ef-search  --k 10 --num-concurrency  1,5,10,20,30,40,60,80 --concurrency-duration 30 --case-type Performance768D1M --num-concurrency 1
vectordbbench milvusivfsq8  --uri http://172.29.0.1:19530 --user-name  --password  --lists 4096 --probes 128 --case-type Performance768D1M --k 10 --num-concurrency 1 --concurrency-duration 30 
# 第一次跑时会创建集合加载数据，后续如果再次测试像直接检索的话，可以添加参数  --skip-drop-old --skip-load  避免重复创建集合
