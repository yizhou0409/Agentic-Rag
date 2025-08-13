#!/usr/bin/env python3
"""
E5-based retriever for semantic search.
"""

import os
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class E5Retriever:
    """E5-based retriever for semantic search."""
    
    def __init__(self, index_path: str, model_name: str = "intfloat/e5-large-v2", device: str = None):
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load E5 model
        logger.info(f"Loading E5 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Load index and metadata
        logger.info(f"Loading E5 index from: {index_path}")
        self.index, self.chunks = self._load_index()
        
        logger.info(f"E5 retriever initialized with {len(self.chunks)} chunks")
    
    def _load_index(self) -> tuple:
        """Load the saved index and metadata."""
        index_path = self.index_path / "index" / "wikipedia_index.faiss"
        metadata_path = self.index_path / "metadata" / "chunks.pkl"
        
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Index or metadata files not found: {index_path}, {metadata_path}")
        
        # Load index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            chunks = pickle.load(f)
            
        return index, chunks
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using E5 model."""
        # According to E5 documentation, queries should use "query: " prefix
        if not text.startswith("query: "):
            text = f"query: {text}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.cpu().numpy()
            
            # Normalize
            faiss.normalize_L2(embedding)
            
        return embedding
    
    def search(self, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), num)
        
        # Get results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                # Ensure the result has a 'text' field for compatibility
                if 'text' not in result:
                    result['text'] = result.get('content', str(result))
                results.append(result)
        
        return results


# Test function
def test_e5_retriever():
    """Test the E5 retriever."""
    print("=== Testing E5 Retriever ===")
    
    # Test with a sample index path (you'll need to adjust this)
    index_path = "indexes/e5"
    
    try:
        retriever = E5Retriever(index_path)
        
        # Test queries
        test_queries = [
            "how many episodes of touching evil are there",
            "who is the owner of reading football club"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.search(query, num=3)
            
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result['score']:.4f}")
                print(f"     Text preview: {result['text'][:200]}...")
                
    except FileNotFoundError as e:
        print(f"Index not found: {e}")
        print("Please ensure the E5 index is properly set up at the specified path.")
    except Exception as e:
        print(f"Error testing E5 retriever: {e}")


if __name__ == "__main__":
    test_e5_retriever()
