#!/usr/bin/env python3
"""
BM25-based retriever using FlashRAG.
"""

import os
import logging
from typing import List, Dict, Any
from flashrag.utils import get_retriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based retriever using FlashRAG backend."""
    
    def __init__(self, index_path: str, top_k_docs: int = 10):
        self.index_path = index_path
        self.top_k_docs = top_k_docs
        
        # Create FlashRAG config for BM25
        retriever_config = {
            "retrieval_method": "bm25",
            "index_path": self.index_path,
            "corpus_path": os.path.join(self.index_path, "corpus.jsonl"),
            "bm25_backend": "bm25s",
            "retrieval_topk": self.top_k_docs,
            "retrieval_batch_size": 32,
            "retrieval_use_fp16": False,
            "retrieval_query_max_length": 128,
            "use_sentence_transformer": True,
            "faiss_gpu": True,
            "silent": False,
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_cache_path": None,
            "use_reranker": False
        }
        
        # Initialize FlashRAG retriever
        logger.info(f"Initializing BM25 retriever with index path: {index_path}")
        self.retriever = get_retriever(retriever_config)
        logger.info("BM25 retriever initialized successfully")
    
    def search(self, query: str, num: int = None) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25.
        
        Args:
            query: Search query
            num: Number of documents to retrieve (overrides top_k_docs if provided)
            
        Returns:
            List of retrieved documents with scores and metadata
        """
        # Use provided num or default top_k_docs
        k = num if num is not None else self.top_k_docs
        
        # Perform search using FlashRAG
        results = self.retriever.search(query, k)
        
        # Convert results to standard format
        formatted_results = []
        for i, result in enumerate(results):
            # Ensure the result has a 'text' field for compatibility
            if isinstance(result, dict):
                if 'text' not in result:
                    result['text'] = result.get('contents', result.get('content', str(result)))
                formatted_results.append(result)
            else:
                # If result is not a dict, create a dict with the text
                formatted_results.append({
                    'text': str(result),
                    'score': 0.0,  # Default score
                    'rank': i + 1
                })
        
        return formatted_results


# Test function
def test_bm25_retriever():
    """Test the BM25 retriever."""
    print("=== Testing BM25 Retriever ===")
    
    # Test with a sample index path (you'll need to adjust this)
    index_path = "indexes/bm25"
    
    try:
        retriever = BM25Retriever(index_path)
        
        # Test queries
        test_queries = [
            "how many episodes of touching evil are there",
            "who is the owner of reading football club"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.search(query, num=3)
            
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result.get('score', 'N/A')}")
                print(f"     Text preview: {result['text'][:200]}...")
                
    except FileNotFoundError as e:
        print(f"Index not found: {e}")
        print("Please ensure the BM25 index is properly set up at the specified path.")
    except Exception as e:
        print(f"Error testing BM25 retriever: {e}")


if __name__ == "__main__":
    test_bm25_retriever()
