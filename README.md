# Agentic-RAG

## Buiding a corpus

This guide explains how to prepare a Wikipedia corpus for use with the Agentic-RAG system. Follow the steps below to download, extract, index, and (optionally) merge your corpus. You may refer to requirements.txt for environments building the corpus.

### 1. Download and Extract the Wikipedia Dump

Download the English Wikipedia dump (We recommend the June 2021 version):

```bash
wget https://archive.org/download/enwiki-20210620/enwiki-20210620-pages-articles.xml.bz2
```

Use the provided `extract_wiki.py` script to extract the dump into a directory (default: `temp`). This script uses [WikiExtractor](https://github.com/attardi/wikiextractor), which must be installed and available in your Python environment.

```bash
python extract_wiki.py --dump_path enwiki-20210620-pages-articles.xml.bz2 --output_dir temp --num_workers 8
```
- `--dump_path`: Path to the downloaded `.bz2` file
- `--output_dir`: Directory to store extracted files (default: `temp`)
- `--num_workers`: Number of parallel processes (adjust as needed)

### 2. Build the E5 Corpus Index(We recommend e5 index. In the main.py we also support bm25 retrieval)

Run the `build_e5_corpus.py` script to process the extracted files and build the E5 vector index. You can run this on the full corpus or in parallel splits.

### (A) Build the Full Index

```bash
python build_e5_corpus.py --data_dir temp --output_dir wikipedia_e5_index
```
- `--data_dir`: Directory with extracted wiki files (from step 2)
- `--output_dir`: Output directory for the E5 index

### (B) Build by Splits (for Parallel Processing)

To process the corpus in parallel, specify `--split_id` and `--total_splits` (run one process per split):

```bash
python build_e5_corpus.py --data_dir temp --output_dir wikipedia_e5_index --split_id 0 --total_splits 4
python build_e5_corpus.py --data_dir temp --output_dir wikipedia_e5_index --split_id 1 --total_splits 4
python build_e5_corpus.py --data_dir temp --output_dir wikipedia_e5_index --split_id 2 --total_splits 4
python build_e5_corpus.py --data_dir temp --output_dir wikipedia_e5_index --split_id 3 --total_splits 4
```

If you built the index in splits, merge them into a single index using `merge_e5_splits.py`:

```bash
python merge_e5_splits.py --output_dir wikipedia_e5_index --total_splits 4
```
- The merged index will be saved in `wikipedia_e5_index/merged/`

## Running Inference

After building your corpus and index, you can run inference using `main.py`.

### Example: Inference with E5 Retrieval

```bash
python main.py \
  --retriever-type e5 \
  --retriever-index-path wikipedia_e5_index/merged \
  --e5-model-path /path/to/e5-large-v2 \
  --reasoner-model Qwen/Qwen3-32B \
  --summarizer-model Qwen/Qwen3-32B \
  --dataset hotpotqa \
  --output-dir output/search_e5
```
- Adjust `--retriever-type` to shift to bm25 or other retrieval methods
- Adjust `--retriever-index-path` to your merged E5 index directory
- Set `--e5-model-path` to your E5 model location

### Additional Options
- Use `--max-samples` to limit the number of questions processed
- Use `--max-turns` to control the number of reasoning turns
- Use `--save-intermediate` to save intermediate results

See all options with:

```bash
python main.py --help
```

Optional, if need to run the pipeline easier, you can modify and run the bash file in the terminal by: 
```bash
./test.sh
```