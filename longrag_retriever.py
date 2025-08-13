#!/usr/bin/env python3
"""
Persistent LongRAG Retriever that builds and saves embeddings to disk.
"""

import os
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongRAGRetriever:
    """LongRAG retriever that builds and saves embeddings to disk for reuse."""
    
    def __init__(self, dataset_name: str = "nq", model_name: str = "BAAI/bge-large-en-v1.5", 
                 device: str = None, max_corpus_size: int = None, 
                 index_dir: str = "longrag_indexes"):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_corpus_size = max_corpus_size
        self.index_dir = Path(index_dir)
        
        # Create index directory
        self.index_dir.mkdir(exist_ok=True)
        
        # Generate index name based on parameters
        self.index_name = f"{dataset_name}_{model_name.split('/')[-1]}"
        if max_corpus_size:
            self.index_name += f"_max{max_corpus_size}"
        
        # Load BGE model
        logger.info(f"Loading BGE model: {model_name}")
        if os.path.exists(model_name):
            logger.info(f"Loading BGE model from local path: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        else:
            logger.info(f"Loading BGE model from Hugging Face Hub: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Load or build index
        self.corpus_data, self.corpus_embeddings = self._load_or_build_index()
        
        logger.info(f"Persistent LongRAG retriever initialized with {len(self.corpus_data)} corpus items")
    
    def _get_index_paths(self):
        """Get paths for index files."""
        index_path = self.index_dir / f"{self.index_name}_embeddings.npy"
        metadata_path = self.index_dir / f"{self.index_name}_metadata.pkl"
        return index_path, metadata_path
    
    def _load_or_build_index(self):
        """Load existing index or build new one."""
        index_path, metadata_path = self._get_index_paths()
        
        # Check if index exists
        if index_path.exists() and metadata_path.exists():
            logger.info(f"Loading existing index from {index_path}")
            try:
                # Load embeddings
                corpus_embeddings = np.load(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    corpus_data = pickle.load(f)
                
                logger.info(f"Successfully loaded index with {len(corpus_data)} items")
                return corpus_data, corpus_embeddings
                
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                logger.info("Building new index...")
        
        # Build new index
        logger.info("Building new index...")
        corpus_data, corpus_embeddings = self._build_index()
        
        # Save index
        logger.info(f"Saving index to {index_path}")
        np.save(index_path, corpus_embeddings)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(corpus_data, f)
        
        logger.info("Index saved successfully")
        return corpus_data, corpus_embeddings
    
    def _build_index(self):
        """Build index from scratch."""
        # Load LongRAG corpus
        logger.info(f"Loading LongRAG corpus: {self.dataset_name}")
        if self.dataset_name == "nq":
            corpus = load_dataset("TIGER-Lab/LongRAG", "nq_corpus")
        elif self.dataset_name == "hotpot_qa":
            corpus = load_dataset("TIGER-Lab/LongRAG", "hotpot_qa_corpus")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        corpus_items = list(corpus['train'])
        
        # Limit corpus size if specified
        if self.max_corpus_size:
            corpus_items = corpus_items[:self.max_corpus_size]
            logger.info(f"Limited corpus to {len(corpus_items)} items")
        
        # Pre-compute embeddings
        logger.info("Pre-computing corpus embeddings...")
        embeddings = []
        
        for i, item in enumerate(corpus_items):
            if i % 1000 == 0:
                logger.info(f"Computing embeddings: {i}/{len(corpus_items)}")
            
            embedding = self._generate_embedding(item['text'])
            embeddings.append(embedding)
        
        # Stack all embeddings
        corpus_embeddings = np.vstack(embeddings)
        logger.info(f"Pre-computed embeddings shape: {corpus_embeddings.shape}")
        
        return corpus_items, corpus_embeddings
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using BGE model."""
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
            
            # Normalize (BGE embeddings are already normalized)
            faiss.normalize_L2(embedding)
            
        return embedding
    
    def search(self, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """Search for long retrieval units similar to the query."""
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Compute similarities with all corpus embeddings
        similarities = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:num]
        
        # Get results
        results = []
        for i, idx in enumerate(top_indices):
            corpus_item = self.corpus_data[idx]
            result = {
                'text': corpus_item['text'],
                'title': corpus_item.get('title', ''),
                'id': corpus_item.get('id', str(idx)),  # Use index as fallback if 'id' doesn't exist
                'score': float(similarities[idx]),
                'rank': i + 1,
                'length_tokens': len(corpus_item['text'].split())  # Approximate
            }
            results.append(result)
        
        return results
    
    def get_index_info(self):
        """Get information about the current index."""
        index_path, metadata_path = self._get_index_paths()
        
        info = {
            'index_name': self.index_name,
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'corpus_size': len(self.corpus_data),
            'embedding_dim': self.corpus_embeddings.shape[1] if self.corpus_embeddings is not None else None,
            'index_exists': index_path.exists() and metadata_path.exists(),
            'index_path': str(index_path),
            'metadata_path': str(metadata_path)
        }
        
        if index_path.exists():
            info['index_size_mb'] = index_path.stat().st_size / (1024 * 1024)
        
        return info


# Test function
def test_persistent_retriever():
    """Test the persistent LongRAG retriever."""
    print("=== Testing Persistent LongRAG Retriever ===")
    
    # Test with small corpus for quick testing
    retriever = PersistentLongRAGRetriever(
        dataset_name="nq", 
        max_corpus_size=1000,  # Limit for testing
        device="cpu"
    )
    
    # Print index info
    info = retriever.get_index_info()
    print(f"Index info: {info}")
    
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
            print(f"     ID: {result['id']}")
            print(f"     Text preview: {result['text'][:200]}...")


if __name__ == "__main__":
    test_persistent_retriever()
