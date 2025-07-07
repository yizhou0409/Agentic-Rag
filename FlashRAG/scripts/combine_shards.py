import numpy as np
import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine embeddings from shards.")
    parser.add_argument("--shard_dir", type=str)
    parser.add_argument("--hidden_size", type=int)
    args = parser.parse_args()

    shard_files = os.listdir(args.shard_dir)
    shard_files = [os.path.join(args.shard_dir, file) for file in shard_files]
    shard_files = sorted(shard_files)

    allembeddings = []
    for file in tqdm(shard_files):
        embeddings = np.memmap(file, mode="r", dtype=np.float32)
        embeddings = embeddings.reshape(-1, args.hidden_size)
        allembeddings.append(embeddings)
    
    allembeddings = np.concatenate(allembeddings, axis=0)
    
    prefix = shard_files[0].split("/")[-1].split("_")[0]
    output_path = os.path.join(args.shard_dir, f"{prefix}_combined_embeddings.memmap")
    
    memmap = np.memmap(output_path, shape=allembeddings.shape, mode="w+", dtype=allembeddings.dtype)
    length = allembeddings.shape[0]

    # add in batch
    save_batch_size = 100000

    if length > save_batch_size:
        for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
            j = min(i + save_batch_size, length)
            memmap[i:j] = allembeddings[i:j]
    else:
        memmap[:] = allembeddings