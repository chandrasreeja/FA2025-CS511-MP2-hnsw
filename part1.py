import faiss
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

def evaluate_hnsw_vs_lsh():
    # ---- Step 1: Load dataset ----
    data_path = './dataset/P2_sift-128-euclidean.hdf5'  # <-- change filename if needed
    print(f"Loading dataset from {data_path} ...")

    with h5py.File(data_path, 'r') as f:
        print("Available keys:", list(f.keys()))
        # Adjust keys if needed
        if 'train' in f:
            xb = f['train'][:]
        elif 'database' in f:
            xb = f['database'][:]
        else:
            raise ValueError("Couldn't find train/database in HDF5 file.")

        if 'test' in f:
            xq = f['test'][:]
        elif 'query' in f:
            xq = f['query'][:]
        else:
            raise ValueError("Couldn't find test/query in HDF5 file.")

        if 'neighbors' in f:
            gt = f['neighbors'][:]
        elif 'ground_truth' in f:
            gt = f['ground_truth'][:]
        else:
            raise ValueError("Couldn't find ground truth neighbor indices.")

    print(f"Database vectors: {xb.shape}, Query vectors: {xq.shape}")

    # ---- Step 2: Evaluate HNSW ----
    hnsw_results = []
    ef_values = [10, 50, 100, 200]
    print("\nEvaluating HNSW...")
    for ef in ef_values:
        index = faiss.IndexHNSWFlat(xb.shape[1], 32)  # M=32
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = ef
        index.add(xb.astype('float32'))

        start = time.time()
        D, I = index.search(xq.astype('float32'), 1)
        elapsed = time.time() - start

        recall = np.mean(I[:, 0] == gt[:, 0])
        qps = len(xq) / elapsed
        hnsw_results.append((ef, recall, qps))

        print(f"HNSW ef={ef:3d} -> Recall={recall:.4f}, QPS={qps:.2f}")

    # ---- Step 3: Evaluate LSH ----
    lsh_results = []
    nbits_values = [32, 64, 512, 768]
    print("\nEvaluating LSH...")
    for nbits in nbits_values:
        index = faiss.IndexLSH(xb.shape[1], nbits)
        index.add(xb.astype('float32'))

        start = time.time()
        D, I = index.search(xq.astype('float32'), 1)
        elapsed = time.time() - start

        recall = np.mean(I[:, 0] == gt[:, 0])
        qps = len(xq) / elapsed
        lsh_results.append((nbits, recall, qps))

        print(f"LSH nbits={nbits:3d} -> Recall={recall:.4f}, QPS={qps:.2f}")

    # ---- Step 4: Plot QPS vs Recall ----
    
    plt.figure(figsize=(8, 6))

    # Plot HNSW results
    ef_vals, recalls_hnsw, qps_hnsw = zip(*hnsw_results)
    plt.plot(recalls_hnsw, qps_hnsw, 'o-', label='HNSW', color='tab:blue')
    for ef, r, q in zip(ef_vals, recalls_hnsw, qps_hnsw):
        plt.text(r, q, f"ef={ef}", fontsize=8)

    # Plot LSH results
    nbits_vals, recalls_lsh, qps_lsh = zip(*lsh_results)
    plt.plot(recalls_lsh, qps_lsh, 's-', label='LSH', color='tab:orange')
    for nbits, r, q in zip(nbits_vals, recalls_lsh, qps_lsh):
        plt.text(r, q, f"nbits={nbits}", fontsize=8)

    plt.xlabel("1‑Recall@1")        # now on x‑axis
    plt.ylabel("Queries per Second (QPS)")   # now on y‑axis
    plt.title("HNSW vs LSH: Recall vs QPS on SIFT1M")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hnsw_vs_lsh_recall_vs_qps.png", dpi=300)
    plt.show()

    print("\nPlot saved as hnsw_vs_lsh_recall_vs_qps.png")


if __name__ == "__main__":
    evaluate_hnsw_vs_lsh()
