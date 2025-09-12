import h5py
import numpy as np
import argparse
import getpass
import json
import sys
from pymilvus import MilvusClient, DataType

def read_hdf5_data(file_path, train_key, save_path=None):
    try:
        with h5py.File(file_path, "r") as f:
            print("all datasets：", list(f.keys()))
            if train_key not in f:
                print(f"error: not found dataset '{train_key}'")
                return [], ()
            print("select dataset:", train_key)
            train_dataset = f[train_key]
            train_data = np.array(train_dataset, dtype=np.float32)
            data_shape = train_data.shape
            print(f"dataset shape：{train_data.shape}")
            print(f"dataset type：{train_data.dtype}")

            result = []
            for vec in train_data:
                result.append({"vec": vec.tolist()})

            print("show first 5 data: ==============")
            print(result[:5])
            print("==============")

            if save_path is not None:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=None, ensure_ascii=False, separators=(',', ':'))
                    print(f"save data to '{save_path}' success")

            return result, data_shape

    except FileNotFoundError:
        print(f"error: not found file '{file_path}'")
        return [], ()
    except Exception as e:
        print(f"error when read file: {str(e)}")
        return [], ()

def delete_milvus_collection(uri, token, collection_name):
    try:
        client = MilvusClient(uri=uri,token=token)
        print(f"connect Milvus success: {uri}")
        client.drop_collection(collection_name=collection_name)
        print(f"collection '{collection_name}' drop success")
    except Exception as e:
        print(f"error: {str(e)}")

def create_and_insert_milvus_collection(uri, token, dim, metric_type, m, ef, collection_name, batch_size, data=None):
    try:
        client = MilvusClient(uri=uri,token=token)
        print(f"connect Milvus success: {uri}")

        schema = MilvusClient.create_schema(auto_id=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vec", datatype=DataType.FLOAT_VECTOR, dim=dim)

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vec",index_type="HNSW",metric_type=metric_type,params={"m": m, "ef": ef})

        if client.has_collection(collection_name=collection_name):
            print(f"collection '{collection_name}' already exists")
            client.drop_collection(collection_name=collection_name)
            print(f"collection '{collection_name}' drop success")
        client.create_collection(collection_name=collection_name,schema=schema,index_params=index_params)

        client.load_collection(collection_name=collection_name)
        print(f"collection '{collection_name}' create success with HNSW index")

        if data is not None:
            total = len(data)
            if total == 0:
                print("no data to insert")
                return
            total_batches = (total + batch_size - 1) // batch_size
            for i in range(total_batches):
                start = i * batch_size
                end = min(start + batch_size, total)
                batch = data[start:end]
                client.insert(collection_name=collection_name,data=batch)
                print(f"insert {len(batch)} entities, progress: {i+1}/{total_batches}")
    except Exception as e:
        print(f"error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='milvus sift bench', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--milvus_uri', type=str, help='milvus uri, eg: http://localhost:19530')
    parser.add_argument('--milvus_token', type=str, help='milvus token, eg: root:Milvus')
    parser.add_argument('--milvus_collection', type=str, default='sift_bench', help='milvus collection name')
    parser.add_argument('--milvus_insert_batchsize', type=str, default='1000', help='milvus insert batchsize')
    parser.add_argument('--milvus_metric_type', type=str, default='IP', help='milvus metric type: IP|L2|COSINE')
    parser.add_argument('--milvus_HNSW_m', type=str, default='8', help='milvus index params: HNSW m')
    parser.add_argument('--milvus_HNSW_ef', type=str, default='200', help='milvus index params: HNSW ef')

    parser.add_argument('--hdf5_path', type=str, default='/app/sift-128-euclidean.hdf5', help='hdf5 file path')
    parser.add_argument('--hdf5_dataset', type=str, default='train', help='hdf5 file dataset name')
    parser.add_argument('--hdf5_dataset_save_path', type=str, default='/app/testdata/dataset.json', help='hdf5 dataset save path')
    parser.add_argument('--op', type=str, default='readAndInsert', help='mode: readAndInsert | readAndSave | read | deleteCollection')
    args = parser.parse_args()

    if args.op == "readAndInsert":
        if args.milvus_uri is None:
            print("error: milvus_uri must be set, eg: http://localhost:19530")
            sys.exit(1)
        if args.milvus_token is None:
            args.milvus_token = getpass.getpass("please input milvus root password: ")
            args.milvus_token = "root:" + args.milvus_token
        data_list, data_shape =  read_hdf5_data(file_path=args.hdf5_path, train_key=args.hdf5_dataset)
        create_and_insert_milvus_collection(uri=args.milvus_uri, token=args.milvus_token, dim=data_shape[1], metric_type=args.milvus_metric_type, m=int(args.milvus_HNSW_m), ef=int(args.milvus_HNSW_ef), collection_name=args.milvus_collection, batch_size=int(args.milvus_insert_batchsize), data=data_list)
    elif args.op == "readAndSave":
        read_hdf5_data(file_path=args.hdf5_path, train_key=args.hdf5_dataset, save_path=args.hdf5_dataset_save_path)
    elif args.op == "read":
        read_hdf5_data(file_path=args.hdf5_path, train_key=args.hdf5_dataset)
    elif args.op == "deleteCollection":
        if args.milvus_uri is None:
            print("error: milvus_uri must be set, eg: http://localhost:19530")
            sys.exit(1)
        if args.milvus_token is None:
            args.milvus_token = getpass.getpass("please input milvus root password: ")
            args.milvus_token = "root:" + args.milvus_token
        delete_milvus_collection(uri=args.milvus_uri, token=args.milvus_token, collection_name=args.milvus_collection)
    else:
        print("error: invalid op")

    print("all done")
