import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    
    # ---- Step 1: Load the HDF5 dataset ----
    data_path = './dataset/P2_sift-128-euclidean.hdf5'   # <-- change to match your file name
    print(f"Loading HDF5 dataset from {data_path} ...")

    with h5py.File(data_path, 'r') as f:
        # You might need to adjust keys depending on file structure
        print("Keys in HDF5 file:", list(f.keys()))

        # Common dataset names (update if needed)
        if 'train' in f:
            xb = f['train'][:]
        elif 'database' in f:
            xb = f['database'][:]
        else:
            raise ValueError("Could not find 'train' or 'database' dataset in HDF5 file.")

        if 'test' in f:
            xq = f['test'][:]
        elif 'query' in f:
            xq = f['query'][:]
        else:
            raise ValueError("Could not find 'test' or 'query' dataset in HDF5 file.")

    print(f"Database vectors shape: {xb.shape}")
    print(f"Query vectors shape: {xq.shape}")

    # ---- Step 2: Create the HNSW index ----
    d = xb.shape[1]
    M = 16
    efConstruction = 200
    efSearch = 200

    print("Creating HNSW index...")
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch

    # ---- Step 3: Add database vectors ----
    print("Adding database vectors to the index...")
    index.add(xb.astype('float32'))
    print(f"Indexed {index.ntotal} vectors.")

    # ---- Step 4: Perform query with first test vector ----
    print("Running query...")
    query = xq[0:1].astype('float32')
    k = 10
    distances, indices = index.search(query, k)

    # ---- Step 5: Write output ----
    output_path = './output.txt'
    with open(output_path, 'w') as f:
        for idx in indices[0]:
            f.write(str(idx) + '\n')

    print(f"Top 10 nearest neighbor indices written to {output_path}")

if __name__ == "__main__":
    evaluate_hnsw()
