SHARD_ID=$1
NUM_SHARDS=$2

python -m flashrag.retriever.encode_shard \
    --retrieval_method bge \
    --model_path /gpfsnyu/scratch/bc3088/models/bge-large-en-v1.5 \
    --corpus_path ../wiki-20231120-chunk256.jsonl \
    --save_dir ../index/bge-2023 \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --sentence_transformer \
    --faiss_type Flat \
    --save_embedding \
    --shard_id $SHARD_ID \
    --num_shards $NUM_SHARDS | tee ../logs/encode_shard.log