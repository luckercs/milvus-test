import h5py
import numpy as np
import argparse
from pymilvus import MilvusClient, DataType

def read_hdf5_data(file_path, train_key="train", save_path=None):
    try:
        with h5py.File(file_path, "r") as f:
            print("all datasets：", list(f.keys()))
            if train_key not in f:
                print(f"error: not found dataset '{train_key}'")
                return [], ()

            train_dataset = f[train_key]
            train_data = np.array(train_dataset, dtype=np.float32)
            data_shape = train_data.shape
            print(f"dataset shape：{train_data.shape}")
            print(f"dataset type：{train_data.dtype}")

            if save_path is not None:
                np.save(save_path, train_data)

            result = []
            for vec in train_data:
                result.append({"vec": vec.tolist()})

            return result, data_shape

    except FileNotFoundError:
        print(f"error: not found file '{file_path}'")
        return [], ()
    except Exception as e:
        print(f"error when read file: {str(e)}")
        return [], ()


def create_milvus_collection(uri, token, dim, metric_type="IP", m=8, ef=200, collection_name="sift_bench", data=None):
    try:
        client = MilvusClient(uri=uri,token=token)

        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",index_type="HNSW",metric_type=metric_type,params={"m": m, "ef": ef})

        client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)

        client.load_collection(collection_name=collection_name)
        print(f"collection '{collection_name}' create success with HNSW index")

        insert_result = None
        if data is not None:
            insert_result = client.insert(collection_name=collection_name,data=data)
            print(f"insert {insert_result.insert_count} entities")

    except Exception as e:
        print(f"error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='milvus sift bench')
    parser.add_argument('--milvus_uri', type=str, default='http://localhost:19530', help='milvus uri')
    parser.add_argument('--milvus_token', type=str, default='root:Milvus', help='milvus token')
    parser.add_argument('--milvus_collection', type=str, default='sift_bench', help='milvus collection name')
    parser.add_argument('--milvus_metric_type', type=str, default='IP', help='milvus metric type: IP|L2|COSINE')
    parser.add_argument('--milvus_HNSW_m', type=str, default='8', help='milvus index params: HNSW m')
    parser.add_argument('--milvus_HNSW_ef', type=str, default='200', help='milvus index params: HNSW ef')

    parser.add_argument('--hdf5_path', type=str, default='./sift-128-euclidean.hdf5', help='hdf5 file path')
    parser.add_argument('--hdf5_dataset', type=str, default='train', help='hdf5 file dataset name')
    args = parser.parse_args()

    data_list, data_shape =  read_hdf5_data(args.hdf5_path, args.hdf5_dataset)
    print(data_list[:5])
    print(data_shape)
    read_hdf5_data(args.hdf5_path, "test", "/app/test.data")
