#!/usr/bin/env python3
"""
Merge script to combine two separate E5 index splits into a single combined index.
This script loads the FAISS indexes and metadata from both splits and merges them.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import pickle
import numpy as np
import faiss
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class E5SplitMerger:
    """Merge multiple E5 index splits into a single combined index."""
    
    def __init__(self, output_dir: str, total_splits: int = 2):
        self.output_dir = Path(output_dir)
        self.total_splits = total_splits
        
        # Create final output directory
        self.final_output_dir = self.output_dir / "merged"
        self.final_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.final_output_dir / "embeddings").mkdir(exist_ok=True)
        (self.final_output_dir / "index").mkdir(exist_ok=True)
        (self.final_output_dir / "metadata").mkdir(exist_ok=True)
        
    def load_split_data(self, split_id: int) -> tuple:
        """Load FAISS index and metadata from a specific split."""
        split_dir = self.output_dir / f"split_{split_id}"
        
        # Load index
        index_path = split_dir / "index" / "wikipedia_index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found for split {split_id}: {index_path}")
        
        logger.info(f"Loading index from split {split_id}...")
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = split_dir / "metadata" / "chunks.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for split {split_id}: {metadata_path}")
        
        logger.info(f"Loading metadata from split {split_id}...")
        with open(metadata_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Load info
        info_path = split_dir / "metadata" / "index_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
        else:
            info = {}
        
        logger.info(f"Split {split_id}: {len(chunks)} chunks, {index.ntotal} vectors")
        return index, chunks, info
    
    def merge_splits(self) -> tuple:
        """Merge all splits into a single combined index."""
        logger.info(f"Starting merge of {self.total_splits} splits...")
        
        # Load first split
        combined_index, combined_chunks, combined_info = self.load_split_data(0)
        
        # Merge additional splits
        for split_id in range(1, self.total_splits):
            logger.info(f"Merging split {split_id}...")
            
            # Load split data
            split_index, split_chunks, split_info = self.load_split_data(split_id)
            
            # Get vectors from split index
            split_vectors = split_index.reconstruct_n(0, split_index.ntotal)
            
            # Add vectors to combined index
            combined_index.add(split_vectors.astype('float32'))
            
            # Extend chunks
            combined_chunks.extend(split_chunks)
            
            # Update info
            if 'num_vectors' in split_info:
                combined_info['num_vectors'] = combined_info.get('num_vectors', 0) + split_info['num_vectors']
            if 'num_batches' in split_info:
                combined_info['num_batches'] = combined_info.get('num_batches', 0) + split_info['num_batches']
        
        logger.info(f"Merged index: {len(combined_chunks)} chunks, {combined_index.ntotal} vectors")
        return combined_index, combined_chunks, combined_info
    
    def save_merged_index(self, index, chunks: List[Dict[str, Any]], info: Dict[str, Any]):
        """Save the merged index and metadata."""
        logger.info("Saving merged index...")
        
        # Save index
        index_path = self.final_output_dir / "index" / "wikipedia_index.faiss"
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved merged index to {index_path}")
        
        # Save metadata
        metadata_path = self.final_output_dir / "metadata" / "chunks.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved merged metadata to {metadata_path}")
        
        # Save info
        info['num_vectors'] = len(chunks)
        info['dimension'] = index.d
        info['index_type'] = 'faiss'
        info['model_name'] = 'e5-large-v2'
        info['num_splits_merged'] = self.total_splits
        
        info_path = self.final_output_dir / "metadata" / "index_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved merged info to {info_path}")
    
    def validate_merge(self, index, chunks: List[Dict[str, Any]]):
        """Validate the merged index."""
        logger.info("Validating merged index...")
        
        # Check vector count
        expected_vectors = len(chunks)
        actual_vectors = index.ntotal
        
        if expected_vectors != actual_vectors:
            logger.warning(f"Vector count mismatch: expected {expected_vectors}, got {actual_vectors}")
        else:
            logger.info(f"Vector count validation passed: {actual_vectors} vectors")
        
        # Check chunk count
        logger.info(f"Total chunks: {len(chunks)}")
        
        # Check split distribution
        split_counts = {}
        for chunk in chunks:
            split_id = chunk.get('split_id', 'unknown')
            split_counts[split_id] = split_counts.get(split_id, 0) + 1
        
        logger.info("Chunk distribution by split:")
        for split_id, count in split_counts.items():
            logger.info(f"  Split {split_id}: {count} chunks")
        
        return True
    
    def create_search_test(self, index, chunks: List[Dict[str, Any]]):
        """Create a simple search test to verify the merged index works."""
        logger.info("Testing merged index with sample search...")
        
        try:
            # Create a simple test query embedding (random for testing)
            test_embedding = np.random.randn(1, index.d).astype('float32')
            faiss.normalize_L2(test_embedding)
            
            # Search
            k = min(5, len(chunks))
            scores, indices = index.search(test_embedding, k)
            
            logger.info(f"Search test successful: found {len(indices[0])} results")
            logger.info(f"Top scores: {scores[0][:3]}")
            
            return True
        except Exception as e:
            logger.error(f"Search test failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Merge E5 index splits")
    parser.add_argument("--output_dir", default="wikipedia_e5_index", 
                       help="Output directory containing splits")
    parser.add_argument("--total_splits", type=int, default=2,
                       help="Total number of splits to merge")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the merged index")
    parser.add_argument("--test_search", action="store_true",
                       help="Test search functionality")
    
    args = parser.parse_args()
    
    # Check if splits exist
    for split_id in range(args.total_splits):
        split_dir = Path(args.output_dir) / f"split_{split_id}"
        if not split_dir.exists():
            logger.error(f"Split directory not found: {split_dir}")
            return
    
    # Initialize merger
    merger = E5SplitMerger(args.output_dir, args.total_splits)
    
    # Merge splits
    start_time = time.time()
    combined_index, combined_chunks, combined_info = merger.merge_splits()
    merge_time = time.time() - start_time
    
    logger.info(f"Merge completed in {merge_time:.2f} seconds")
    
    # Save merged index
    merger.save_merged_index(combined_index, combined_chunks, combined_info)
    
    # Validate if requested
    if args.validate:
        merger.validate_merge(combined_index, combined_chunks)
    
    # Test search if requested
    if args.test_search:
        merger.create_search_test(combined_index, combined_chunks)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Splits merged: {args.total_splits}")
    logger.info(f"Total chunks: {len(combined_chunks)}")
    logger.info(f"Total vectors: {combined_index.ntotal}")
    logger.info(f"Vector dimension: {combined_index.d}")
    logger.info(f"Merge time: {merge_time:.2f} seconds")
    logger.info(f"Final output: {merger.final_output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
