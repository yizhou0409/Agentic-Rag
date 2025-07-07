from flashrag.retriever.index_builder import Index_Builder
import argparse

def main():
    parser = argparse.ArgumentParser(description="Creating index.")

    # Basic parameters
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)

    # Parameters for building dense index
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true", default=False)

    # Parameters for build multi-modal retriever index
    parser.add_argument("--index_modal", type=str, default="all", choices=["text", "image", "all"])

    # Parameters for sharding
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=args.pooling_method,
        instruction=args.instruction,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        use_sentence_transformer=args.sentence_transformer,
        index_modal=args.index_modal,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    hidden_size = index_builder.load_encoder()
    
    all_embeddings = index_builder.encode_all_clip() if index_builder.is_clip else index_builder.encode_all()
    index_builder._save_embedding(all_embeddings)

if __name__ == "__main__":
    main()
