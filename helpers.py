import numpy as np
import h5py
import os
import random

import hnswlib
from annoy import AnnoyIndex
from pyflann import FLANN
import nmslib

from typing import Tuple, List, Union

def get_dataset_fn(dataset_name: str) -> str:
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")

def get_dataset(dataset_name: str) -> Tuple[h5py.File, int]:
    hdf5_filename = get_dataset_fn(dataset_name)
    hdf5_file = h5py.File(hdf5_filename, "r")

    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension

def convert_sparse_to_list(data: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    return [
        data[i - l : i] for i, l in zip(np.cumsum(lengths), lengths)
    ]

def dataset_transform(dataset: h5py.Dataset, dataset_name: str) -> Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    
    train, test = np.array(dataset["train"]), np.array(dataset["test"])

    if dataset_name == 'deep-image-96-euclidean':
        subset_size = 1000000
        train = np.array(random.choices(dataset["train"], k=subset_size))
    
    if dataset.attrs.get("type", "dense") != "sparse":
        return train, test
    # we store the dataset as a list of integers, accompanied by a list of lengths in hdf5
    # so we transform it back to the format expected by the algorithms here (array of array of ints)
    train = convert_sparse_to_list(dataset["train"], dataset["size_train"])
    test = convert_sparse_to_list(dataset["test"], dataset["size_test"])

    return train, test

def load_and_transform_dataset(dataset_name: str) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Union[np.ndarray, List[np.ndarray]],
        str]:

    D, dimension = get_dataset(dataset_name)

    train, test = dataset_transform(D, dataset_name)
    print(f"Got a train set of size ({train.shape[0]} * {dimension})")
    print(f"Got {len(test)} queries")

    return train, test, dimension
#===============================================================
def initialize_brute(dim, metric):
    brute_index = hnswlib.BFIndex(space=metric, dim=dim)
    return brute_index

def build_brute(data, dim, metric): 
    if metric == 'angular':
        metric = 'cosine'
    brute_index = initialize_brute(dim, metric)
    brute_index.init_index(max_elements=len(data))
    brute_index.add_items(data)
    return brute_index

def run_brute(algo, query_data, k):
    labels, distances = algo.knn_query(query_data, k)
    return labels, distances

def brute_export_index(algo, filename, result_dir):
    algo.save_index(f"{result_dir}/{filename}.bin")

def brute_load_index(filename, dim, metric, result_dir):
    if metric == 'angular':
        metric = 'cosine'
    brute_index = initialize_brute(dim, metric)
    brute_index.load_index(f"{result_dir}/{filename}.bin")

    return brute_index

#====================================================
# HNSW
def initialize_hnsw(dim, metric):
    hnsw_index = hnswlib.Index(space=metric, dim=dim)
    return hnsw_index

def build_hnsw(data, dim, metric):
    hnsw_index = initialize_hnsw(dim, metric) # hnswlib only support l2 distance
    hnsw_index.init_index(max_elements=len(data))
    hnsw_index.add_items(data)

    return hnsw_index

def run_hnsw(algo, query_data, k):
    if query_data.dtype != np.float32:
        query_data = query_data.astype(np.float32)
    labels, distances = algo.knn_query(query_data, k)

    return labels, distances

def hnsw_export_index(algo, filename, result_dir):
    algo.save_index(f"{result_dir}/{filename}.bin")

def hnsw_load_index(filename, dim, metric, result_dir):
    hnsw_index = initialize_hnsw(dim, metric)
    hnsw_index.load_index(f"{result_dir}/{filename}.bin")

    return hnsw_index
#============================================
# Annoy
def initialize_annoy(dim, metric):
    if metric == 'l2':
        metric = 'euclidean'
    annoy_index = AnnoyIndex(dim, metric)
    return annoy_index

def build_annoy(data, dim, n_trees, metric, n_jobs=-1):
    annoy_index = initialize_annoy(dim, metric)
    for idx, point in enumerate(data):
        annoy_index.add_item(idx, point)
    
    annoy_index.build(n_trees, n_jobs)

    return annoy_index

def run_annoy(algo, query_data, k):
    annoy_result = []
    for v in query_data:
        annoy_result.append(np.array(algo.get_nns_by_vector(v, k, include_distances=True)))

    return annoy_result

def transform_annoy_result(annoy_result):
    # don't include in query time count
    # this is used to transform the result into similar formats of other libaries
    annoy_labels = [res[0] for res in annoy_result]
    annoy_dist = [res[1] for res in annoy_result]

    return annoy_labels, annoy_dist

def annoy_export_index(algo, filename, result_dir):
    algo.save(f'{result_dir}/{filename}.ann')

def annoy_load_index(filename, dim, metric, result_dir):
    annoy_index = initialize_annoy(dim, metric)
    annoy_index.load(f'{result_dir}/{filename}.ann') # super fast, will just map the file
    return annoy_index 

#=======================================

def initialize_flann():
    flann_index = FLANN(algorithm="autotuned", target_precision=0.95, build_weight=0.5, log_level="info")
    return flann_index

def build_flann(data, metric):
    flann_index = initialize_flann()
    flann_index.build_index(data)
    return flann_index

def run_flann(algo, query_data, k, metric):
    if query_data.dtype != np.float32:
        query_data = query_data.astype(np.float32)
    labels, distances = algo.nn_index(query_data, num_neighbors=k)
    return labels, distances

def flann_export_index(algo, filename, result_dir):
    algo.save_index(f"{result_dir}/{filename}".encode('utf-8'))

def flann_load_index(filename, pts, result_dir):
    flann_index = initialize_flann()
    flann_index.load_index(f"{result_dir}/{filename}".encode('utf-8'), pts)
    return flann_index
#====================================================
def recall(aknn_labels, bf_labels):
    # aknn_labels : approximate knn result
    # bf_labels : bruteforce result
    # result is the labels/neighbors returned

    percentages = []
    for i in range(len(aknn_labels)):
        percentage = (np.isin(aknn_labels[i], bf_labels[i]).sum() / bf_labels[i].size)
        percentages.append(percentage)

    return np.array(percentages)

# =================================================
def export_to_hdf5(arr, filename, dataset_name, result_dir):
    with h5py.File(f"{result_dir}/{filename}.hdf5", 'w') as f:
        f.create_dataset(dataset_name, data=arr)

def load_exported_results(file_name, result_dir):
    # pake ini buat ambil balik hasil yg ke export, g perlu di filepath data langsung selevel sama ipynb
    hdf5_file = h5py.File(f"{result_dir}/{file_name}.hdf5", "r")
    return np.array(hdf5_file[file_name])
# ===================================================