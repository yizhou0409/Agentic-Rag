#!/usr/bin/env python3
"""
Retriever for searchr1_e5: E5-based retriever using custom index and corpus structure.
"""

import os
import json
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

class SearchR1E5Retriever:
    """Retriever for searchr1_e5: E5-based retriever using custom index and corpus structure."""
    def __init__(self, index_dir: str, model_name: str = "intfloat/e5-large-v2", device: str = None):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load E5 model
        logger.info(f"Loading E5 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Load index and corpus
        logger.info(f"Loading FAISS index and corpus from: {index_dir}")
        self.index, self.chunks = self._load_index_and_corpus()
        logger.info(f"searchr1_e5 retriever initialized with {len(self.chunks)} chunks")

    def _load_index_and_corpus(self) -> tuple:
        """Load the FAISS index and the corpus as metadata."""
        index_path = self.index_dir / "wiki-18-e5-index" / "e5_Flat.index"
        corpus_path = self.index_dir / "wiki-18-corpus" / "wiki-18.jsonl"

        if not index_path.exists() or not corpus_path.exists():
            raise FileNotFoundError(f"Index or corpus files not found: {index_path}, {corpus_path}")

        # Load FAISS index
        index = faiss.read_index(str(index_path))

        # Load corpus (as list of dicts)
        chunks = []
        with open(corpus_path, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    # Ensure 'text' field for compatibility
                    if 'text' not in obj:
                        obj['text'] = obj.get('contents', str(obj))
                    chunks.append(obj)
                except Exception as e:
                    logger.warning(f"Skipping line due to error: {e}")
        return index, chunks

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using E5 model."""
        if not text.startswith("query: "):
            text = f"query: {text}"
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = embedding.cpu().numpy()
            faiss.normalize_L2(embedding)
        return embedding

    def search(self, query: str, num: int = 10) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        query_embedding = self._generate_embedding(query)
        scores, indices = self.index.search(query_embedding.astype('float32'), num)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['score'] = float(score)
                result['rank'] = i + 1
                if 'text' not in result:
                    result['text'] = result.get('contents', str(result))
                results.append(result)
        return results

# Test function
def test_searchr1_e5_retriever():
    print("=== Testing searchr1_e5 Retriever ===")
    index_dir = "indexes/searchr1_e5"
    try:
        retriever = SearchR1E5Retriever(index_dir)
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
        print(f"Index or corpus not found: {e}")
        print("Please ensure the searchr1_e5 index and corpus are properly set up at the specified path.")
    except Exception as e:
        print(f"Error testing searchr1_e5 retriever: {e}")

if __name__ == "__main__":
    test_searchr1_e5_retriever()
