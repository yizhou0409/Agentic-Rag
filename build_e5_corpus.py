#!/usr/bin/env python3
"""
Build Wikipedia corpus from bz2 files with correct text extraction.
This script processes Wikipedia data and creates a corpus for E5 retrieval.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
import pickle
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import gc
import time
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikipediaCorpusBuilder:
    """Build Wikipedia corpus with correct text extraction."""
    
    def __init__(self, data_dir: str, output_dir: str, split_id: int = None, total_splits: int = 1):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.split_id = split_id
        self.total_splits = total_splits
        
        # Create output directory
        if split_id is not None:
            self.output_dir = self.output_dir / f"split_{split_id}"
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "index").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
    def get_processed_files(self) -> set:
        """Get list of already processed files."""
        checkpoint_file = self.output_dir / "checkpoints" / "processed_files.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def save_processed_files(self, processed_files: set):
        """Save list of processed files."""
        checkpoint_file = self.output_dir / "checkpoints" / "processed_files.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(list(processed_files), f)
    
    def get_wiki_files(self) -> List[Path]:
        """Get wiki_* files to process from temp/ directory."""
        all_files = []
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                all_files.extend(subdir.glob("wiki_*"))
        
        all_files = sorted(all_files)
        logger.info(f"Found {len(all_files)} total wiki files")
        
        # Split files if needed
        if self.split_id is not None and self.total_splits > 1:
            files_per_split = math.ceil(len(all_files) / self.total_splits)
            start_idx = self.split_id * files_per_split
            end_idx = min(start_idx + files_per_split, len(all_files))
            
            split_files = all_files[start_idx:end_idx]
            logger.info(f"Split {self.split_id}: processing {len(split_files)} files (files {start_idx}-{end_idx-1})")
            return split_files
        else:
            return all_files
    
    def extract_text_from_structure(self, text_data: List) -> str:
        """Extract text from the Wikipedia text structure."""
        if not isinstance(text_data, list):
            return str(text_data)
        
        # Skip the first element (title)
        if len(text_data) <= 1:
            return ""
        
        # Process paragraphs (elements 1 onwards)
        paragraphs = []
        for paragraph_list in text_data[1:]:
            if isinstance(paragraph_list, list):
                # Join all strings in this paragraph
                paragraph_text = " ".join([str(item) for item in paragraph_list if item])
                if paragraph_text.strip():
                    paragraphs.append(paragraph_text)
        
        # Join all paragraphs
        full_text = " ".join(paragraphs)
        return full_text.strip()
    
    def extract_articles_from_file(self, wiki_file: Path, max_articles: int = None) -> List[Dict[str, Any]]:
        """Extract articles from a single wiki file (JSON lines format)."""
        articles = []
        article_count = 0
        
        try:
            with open(wiki_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if max_articles and article_count >= max_articles:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        
                        if 'id' in data and 'title' in data and 'text' in data:
                            # Text is already a string in JSON lines format
                            article_text = data['text']
                            
                            if 50 <= len(article_text) <= 100000:  # Increased max length
                                article = {
                                    'id': data['id'],
                                    'title': data['title'],
                                    'url': data.get('url', ''),
                                    'text': article_text,
                                    'revid': data.get('revid', '')
                                }
                                
                                articles.append(article)
                                article_count += 1
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading {wiki_file}: {e}")
            
        return articles
    
    def create_chunks(self, articles: List[Dict[str, Any]], chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Create overlapping chunks from articles for better retrieval."""
        chunks = []
        chunk_id = 0
        
        for article in articles:
            text = article['text']
            
            # Split text into chunks
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                    
                chunk = {
                    'chunk_id': f"{article['id']}_{chunk_id}",
                    'article_id': article['id'],
                    'article_title': article['title'],
                    'article_url': article['url'],
                    'text': chunk_text,
                    'chunk_index': i // (chunk_size - overlap),
                    'total_chunks': (len(words) + chunk_size - overlap - 1) // (chunk_size - overlap),
                    'split_id': self.split_id if self.split_id is not None else 0
                }
                
                chunks.append(chunk)
                chunk_id += 1
                
        return chunks

class E5Embedder:
    """Generate embeddings using E5 model."""
    
    def __init__(self, model_name: str = "/scratch/yl9038/models/e5-large-v2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading E5 model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model directly from local path
        if os.path.exists(model_name):
            logger.info(f"Loading model directly from: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                logger.info("E5 model loaded from local path successfully")
            except Exception as e:
                logger.error(f"Failed to load from local path: {e}")
                raise
        else:
            logger.error(f"Model path not found: {model_name}")
            raise FileNotFoundError(f"Model path not found: {model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("E5 model loaded successfully")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Add prefix for E5
            batch_texts = [f"passage: {text}" for text in batch_texts]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                attention_mask = inputs['attention_mask']
                embeddings_batch = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                embeddings.append(embeddings_batch.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Perform mean pooling on token embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class VectorDatabase:
    """Manage vector database."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.index_dir = output_dir / "index"
        self.metadata_dir = output_dir / "metadata"
        
    def load_or_create_index(self, dimension: int = 1024):
        """Load existing index or create new one."""
        index_path = self.index_dir / "wikipedia_index.faiss"
        
        if index_path.exists():
            logger.info("Loading existing FAISS index...")
            return faiss.read_index(str(index_path))
        else:
            logger.info("Creating new FAISS index...")
            return faiss.IndexFlatIP(dimension)
    
    def save_index(self, index, chunks: List[Dict[str, Any]], batch_num: int):
        """Save index and metadata."""
        # Save index
        index_path = self.index_dir / "wikipedia_index.faiss"
        faiss.write_index(index, str(index_path))
        
        # Save metadata incrementally
        metadata_path = self.metadata_dir / f"chunks_batch_{batch_num}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        # Update combined metadata
        self._update_combined_metadata(batch_num)
        
        logger.info(f"Saved batch {batch_num} with {len(chunks)} chunks")
    
    def _update_combined_metadata(self, batch_num: int):
        """Update the combined metadata file."""
        combined_chunks = []
        
        # Load all batch metadata
        for i in range(batch_num + 1):
            batch_path = self.metadata_dir / f"chunks_batch_{i}.pkl"
            if batch_path.exists():
                with open(batch_path, 'rb') as f:
                    batch_chunks = pickle.load(f)
                    combined_chunks.extend(batch_chunks)
        
        # Save combined metadata
        combined_path = self.metadata_dir / "chunks.pkl"
        with open(combined_path, 'wb') as f:
            pickle.dump(combined_chunks, f)
        
        # Update info
        info_path = self.metadata_dir / "index_info.json"
        info = {
            'num_vectors': len(combined_chunks),
            'dimension': 1024,
            'index_type': 'faiss',
            'model_name': 'e5-large-v2',
            'num_batches': batch_num + 1
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Build Wikipedia corpus")
    parser.add_argument("--data_dir", default="enwiki-20171001-pages-meta-current-withlinks-processed", 
                       help="Path to Wikipedia data directory")
    parser.add_argument("--output_dir", default="wikipedia_e5_index", 
                       help="Output directory for processed data")
    parser.add_argument("--split_id", type=int, default=None,
                       help="Split ID (for parallel processing)")
    parser.add_argument("--total_splits", type=int, default=1,
                       help="Total number of splits")
    parser.add_argument("--batch_size", type=int, default=1000, 
                       help="Number of articles to process per batch")
    parser.add_argument("--embedding_batch_size", type=int, default=32, 
                       help="Batch size for embedding generation")
    parser.add_argument("--chunk_size", type=int, default=512, 
                       help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=50, 
                       help="Overlap between chunks")
    parser.add_argument("--model_name", default="/scratch/yl9038/models/e5-large-v2", 
                       help="E5 model path")
    parser.add_argument("--device", default=None, 
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from last checkpoint")
    
    args = parser.parse_args()
    
    # Initialize components
    builder = WikipediaCorpusBuilder(args.data_dir, args.output_dir, args.split_id, args.total_splits)
    embedder = E5Embedder(args.model_name, args.device)
    vector_db = VectorDatabase(builder.output_dir)
    
    # Get files and processed files
    wiki_files = builder.get_wiki_files()
    processed_files = builder.get_processed_files() if args.resume else set()
    
    split_info = f"Split {args.split_id}" if args.split_id is not None else "Full dataset"
    logger.info(f"{split_info}: Total files to process: {len(wiki_files)}")
    logger.info(f"{split_info}: Already processed: {len(processed_files)}")
    logger.info(f"{split_info}: Remaining: {len(wiki_files) - len(processed_files)}")
    
    # Load or create index
    index = vector_db.load_or_create_index()
    batch_num = len(processed_files) // args.batch_size
    
    # Process files in batches
    total_files = len(wiki_files)
    start_time = time.time()
    
    # Accumulate chunks and embeddings for batch saving
    accumulated_chunks = []
    accumulated_embeddings = []
    save_every_n_batches = 1000  # Save every 1000 batches
    
    for i, wiki_file in enumerate(tqdm(wiki_files, desc=f"Processing {split_info}")):
        if str(wiki_file) in processed_files:
            continue
            
        try:
            # Extract articles from file
            articles = builder.extract_articles_from_file(wiki_file)
            
            if not articles:
                if i < 5:  # Only log for first few files
                    logger.info(f"No articles found in {wiki_file.name}")
                processed_files.add(str(wiki_file))
                builder.save_processed_files(processed_files)
                continue
            
            logger.info(f"Processing file {i+1}/{len(wiki_files)}: {wiki_file.name} - {len(articles)} articles")
            
            # Create chunks
            chunks = builder.create_chunks(articles, args.chunk_size, args.chunk_overlap)
            
            if not chunks:
                processed_files.add(str(wiki_file))
                builder.save_processed_files(processed_files)
                continue
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedder.generate_embeddings_batch(texts, args.embedding_batch_size)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings)
            
            # Accumulate chunks and embeddings
            accumulated_chunks.extend(chunks)
            accumulated_embeddings.append(embeddings)
            
            # Add to index
            index.add(embeddings.astype('float32'))
            
            # Mark as processed
            processed_files.add(str(wiki_file))
            
            # Save checkpoint every N batches
            if len(accumulated_embeddings) >= save_every_n_batches:
                logger.info(f"Saving checkpoint after {len(accumulated_embeddings)} batches...")
                
                # Combine accumulated embeddings
                combined_embeddings = np.vstack(accumulated_embeddings)
                
                # Save batch
                vector_db.save_index(index, accumulated_chunks, batch_num)
                batch_num += 1
                
                # Save processed files
                builder.save_processed_files(processed_files)
                
                # Clear accumulated data
                accumulated_chunks = []
                accumulated_embeddings = []
                
                # Clear memory
                del combined_embeddings
                gc.collect()
                torch.cuda.empty_cache()
            
            # Clear memory for current batch
            del articles, chunks, texts, embeddings
            gc.collect()
            torch.cuda.empty_cache()
            
            # Log progress with time estimation
            if (i + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                files_per_second = (i + 1) / elapsed_time
                remaining_files = total_files - (i + 1)
                estimated_remaining_time = remaining_files / files_per_second
                estimated_hours = estimated_remaining_time / 3600
                
                logger.info(f"{split_info}: Processed {i + 1}/{total_files} files ({((i+1)/total_files)*100:.1f}%)")
                logger.info(f"{split_info}: Estimated remaining time: {estimated_hours:.1f} hours")
                logger.info(f"{split_info}: Processing rate: {files_per_second:.2f} files/second")
                
        except Exception as e:
            logger.error(f"Error processing {wiki_file}: {e}")
            continue
    
    # Save any remaining accumulated data
    if accumulated_chunks:
        logger.info(f"Saving final {len(accumulated_embeddings)} batches...")
        combined_embeddings = np.vstack(accumulated_embeddings)
        vector_db.save_index(index, accumulated_chunks, batch_num)
        batch_num += 1
        builder.save_processed_files(processed_files)
    
    logger.info(f"{split_info} corpus building completed!")
    logger.info(f"{split_info}: Total files processed: {len(processed_files)}")
    logger.info(f"{split_info}: Total batches: {batch_num}")


if __name__ == "__main__":
    main()
