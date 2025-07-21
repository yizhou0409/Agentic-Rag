import os
import json
import requests

# URLs for the raw files
CORPUS_URL = "https://huggingface.co/datasets/thinkall/2WikiMultihopQA/resolve/main/corpus.jsonl"
QUERIES_URL = "https://huggingface.co/datasets/thinkall/2WikiMultihopQA/resolve/main/queries.jsonl"

save_dir = os.path.join('data', '2wikimultihop')
os.makedirs(save_dir, exist_ok=True)

def download_file(url, out_path):
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {out_path}")

# Download the files
corpus_path = os.path.join(save_dir, "corpus.jsonl")
queries_path = os.path.join(save_dir, "queries.jsonl")
download_file(CORPUS_URL, corpus_path)
download_file(QUERIES_URL, queries_path)

# Split queries.jsonl into train/dev/test
splits = {"train": [], "dev": [], "test": []}
with open(queries_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        split = item.get("type")
        if split in splits:
            splits[split].append(item)

for split, items in splits.items():
    out_path = os.path.join(save_dir, f"{split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(items)} items to {out_path}") 