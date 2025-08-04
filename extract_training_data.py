#!/usr/bin/env python3
"""
Extract training data from datasets and generate consistency labels.
This script:
1. Extracts search queries and contexts from training datasets
2. Uses base 32B model to generate answers
3. Uses judge 32B model to determine consistency
4. Saves results to JSON for training
"""

import os
import json
import torch
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path: str, use_quantization: bool = True, use_multi_gpu: bool = False):
    """Load model with the same approach as inference_consistency_head.py."""
    
    logger.info(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {available_gpus}")
    
    # Configure device map for multi-GPU
    if use_multi_gpu and available_gpus > 1:
        device_map = "auto"
        max_memory = {0: "28GB", 1: "28GB", "cpu": "100GB"}
        logger.info("Using multi-GPU configuration with device_map='auto'")
    else:
        device_map = "auto"
        max_memory = None
        logger.info("Using single GPU configuration")
    
    # Load base model
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    logger.info("Model loaded successfully!")
    return model, tokenizer

def extract_search_context_and_answer(trajectory_text: str) -> List[Dict[str, str]]:
    """
    Extract all search contexts and answers from trajectory text.
    Returns: List of {
        'search_query': str,
        'search_context': str,  # from start to </search>
        'search_answer': str    # from <information> to </information>
    }
    """
    try:
        # Find all search queries
        search_matches = re.finditer(r'<search>(.*?)</search>', trajectory_text, re.DOTALL)
        search_queries = []
        
        for match in search_matches:
            search_query = match.group(1).strip()
            search_queries.append({
                'query': search_query,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        if not search_queries:
            return []
        
        # Find all information blocks
        info_matches = re.finditer(r'<information>(.*?)</information>', trajectory_text, re.DOTALL)
        info_blocks = []
        
        for match in info_matches:
            info_content = match.group(1).strip()
            info_blocks.append({
                'content': info_content,
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        if not info_blocks:
            return []
        
        # Match search queries with their corresponding information blocks
        results = []
        for i, search_query_info in enumerate(search_queries):
            # Find the information block that comes after this search query
            # but before the next search query (if any)
            next_search_start = search_queries[i + 1]['start_pos'] if i + 1 < len(search_queries) else len(trajectory_text)
            
            # Find the first information block that comes after this search query
            corresponding_info = None
            for info_block in info_blocks:
                if (info_block['start_pos'] > search_query_info['end_pos'] and 
                    info_block['start_pos'] < next_search_start):
                    corresponding_info = info_block
                    break
            
            if corresponding_info:
                # Create search context (from start to this search query's </search>)
                search_context = trajectory_text[:search_query_info['end_pos']]
                
                results.append({
                    'search_query': search_query_info['query'],
                    'search_context': search_context,
                    'search_answer': corresponding_info['content']
                })
        
        return results
    except Exception as e:
        logger.error(f"Error extracting from trajectory: {e}")
        return []

def generate_model_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                         search_query: str) -> str:
    """Generate model's answer to search query (without search results)."""
    
    # Create brief prompt with examples for answer generation
    prompt = f"""You are a helpful assistant. Answer questions directly and concisely.

EXAMPLES:

Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: What is 2+2?
Answer: 4

---

Question: {search_query}
Answer:"""
    
    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024,
        padding=False
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate answer
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
            use_cache=False
        )
    
    # Decode
    input_length = len(inputs['input_ids'][0])
    new_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text.strip()

def judge_consistency_with_output(judge_model: AutoModelForCausalLM, judge_tokenizer: AutoTokenizer,
                     search_query: str, search_answer: str, model_answer: str) -> Tuple[float, str]:
    """Use judge model to determine consistency between search answer and model answer and return original output."""
    
    # Create few-shot consistency judgment prompt
    prompt = f"""You are a consistency judge. Your task is to determine if a model's answer is consistent with the search results.

INSTRUCTIONS:
- Focus only on whether the key facts align between the search results and model's answer
- Ignore minor details, formatting differences, or additional information
- Answer with exactly "Consistent" or "Inconsistent"

EXAMPLES:

Question: What is the capital of France?
Search Results: Paris is the capital and largest city of France.
Model's Answer: The capital of France is Paris.
Answer: Consistent

Question: Who wrote Romeo and Juliet?
Search Results: William Shakespeare wrote Romeo and Juliet.
Model's Answer: Romeo and Juliet was written by William Shakespeare.
Answer: Consistent

Question: What is the population of Tokyo?
Search Results: Tokyo has a population of approximately 14 million people.
Model's Answer: Tokyo's population is around 37 million people.
Answer: Inconsistent

Question: When was the Declaration of Independence signed?
Search Results: The Declaration of Independence was signed on July 4, 1776.
Model's Answer: July 4, 1776
Answer: Consistent

Question: What is the largest planet in our solar system?
Search Results: Jupiter is the largest planet in our solar system.
Model's Answer: Jupiter
Answer: Consistent

Question: Who won the 2020 US Presidential election?
Search Results: Joe Biden won the 2020 US Presidential election.
Model's Answer: Donald Trump won the 2020 election.
Answer: Inconsistent

Question: What is the chemical symbol for gold?
Search Results: The chemical symbol for gold is Au.
Model's Answer: Au
Answer: Consistent

Question: How many sides does a hexagon have?
Search Results: A hexagon has six sides.
Model's Answer: 6
Answer: Consistent

Question: Who painted the Mona Lisa?
Search Results: Leonardo da Vinci painted the Mona Lisa.
Model's Answer: Michelangelo painted the Mona Lisa.
Answer: Inconsistent

Question: What is the speed of light?
Search Results: The speed of light is approximately 299,792 kilometers per second.
Model's Answer: About 300,000 km/s
Answer: Inconsistent

Question: How many Grand Slam titles did Henri Leconte win?
Search Results: Henri Leconte won three Grand Slam singles titles: the 1985 French Open, the 1986 French Open, and the 1985 US Open.
Model's Answer: Henri Leconte won 1 Grand Slam title, the 1985 French Open doubles with Yannick Noah. He did not win any Grand Slam singles titles.
Answer: Inconsistent

---

Question: {search_query}
Search Results: {search_answer}
Model's Answer: {model_answer}
Answer:"""
    
    # Tokenize
    inputs = judge_tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,  # Increased for few-shot examples
        padding=False
    )
    
    # Move to device
    device = next(judge_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate judgment
    judge_model.eval()
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,  # Use greedy decoding for judgment
            pad_token_id=judge_tokenizer.eos_token_id,
            use_cache=False
        )
    
    # Decode judgment
    input_length = len(inputs['input_ids'][0])
    new_tokens = outputs[0][input_length:]
    judgment = judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Use all new tokens for consistency determination
    if len(new_tokens) > 0:
        all_tokens_text = judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    else:
        all_tokens_text = ""
    
    # Convert to consistency score based on all tokens
    if "inconsistent" in all_tokens_text:
        return 0.0, judgment
    elif "consistent" in all_tokens_text:
        return 1.0, judgment
    else:
        return -1, judgment

def process_multiple_datasets(dataset_paths: List[str], base_model_path: str, judge_model_path: str,
                            output_path: str, max_examples_per_dataset: int = 100, use_quantization: bool = True,
                            use_multi_gpu: bool = False):
    """Process multiple datasets and generate training examples."""
    
    # Load both models once
    logger.info("Loading base model for answer generation...")
    base_model, base_tokenizer = load_model(base_model_path, use_quantization, use_multi_gpu)
    
    logger.info("Loading judge model for consistency judgment...")
    judge_model, judge_tokenizer = load_model(judge_model_path, use_quantization, use_multi_gpu)
    
    all_training_examples = []
    
    for dataset_path in dataset_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_path}")
        logger.info(f"{'='*60}")
        
        # Load dataset
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")
        
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        
        # Process examples for this dataset
        training_examples = []
        processed_count = 0
        
        # Show progress every 10 examples
        progress_interval = max(1, min(10, len(dataset) // 10))
        
        for i, example in enumerate(dataset):
            if processed_count >= max_examples_per_dataset:
                break
                
            # Extract trajectory text (assuming it's in 'trajectory' field)
            trajectory_text = example.get('trajectory', '')
            if not trajectory_text:
                continue
            
            # Extract all search contexts and answers
            extracted_list = extract_search_context_and_answer(trajectory_text)
            if not extracted_list:
                continue
            
            for extracted in extracted_list:
                search_query = extracted['search_query']
                search_context = extracted['search_context']
                search_answer = extracted['search_answer']
                
                try:
                    # Generate model's answer
                    model_answer = generate_model_answer(base_model, base_tokenizer, search_query)
                    
                    # Judge consistency and get original judgment text
                    consistency_score, judge_output = judge_consistency_with_output(
                        judge_model, judge_tokenizer, 
                        search_query, search_answer, model_answer
                    )

                    if consistency_score == -1:
                        continue
                    
                    # Create training example with judge output
                    training_example = {
                        'dataset': dataset_path.split('/')[-2],  # Extract dataset name
                        'question': example.get('question', ''),  # Save the original question
                        'search_query': search_query,
                        'search_context': search_context,
                        'search_answer': search_answer,
                        'model_answer': model_answer,
                        'judge_output': judge_output,
                        'consistency_label': consistency_score
                    }
                    
                    training_examples.append(training_example)
                    processed_count += 1
                    
                    # Check if we've reached the limit for this dataset
                    if processed_count >= max_examples_per_dataset:
                        break
                
                except Exception as e:
                    logger.error(f"Error processing search query: {e}")
                    continue
            
            # Show progress periodically
            if (i + 1) % progress_interval == 0:
                logger.info(f"Dataset {dataset_path.split('/')[-2]}: Processed {i+1}/{len(dataset)} examples, {processed_count} successful")
            
            # Check if we've reached the limit for this dataset
            if processed_count >= max_examples_per_dataset:
                break
        
        logger.info(f"Completed dataset {dataset_path}: {len(training_examples)} examples processed")
        all_training_examples.extend(training_examples)
    
    # Save combined results
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving {len(all_training_examples)} total training examples to {output_path}")
    logger.info(f"{'='*60}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_training_examples, f, indent=2, ensure_ascii=False)
    
    logger.info("Processing completed!")

def main():
    parser = argparse.ArgumentParser(description="Extract training data and generate consistency labels")
    parser.add_argument("--dataset_paths", type=str, 
                       default="./data/hotpotqa/trajectory_train_200.jsonl,./data/2wikimultihop/trajectory_train_200.jsonl",
                       help="Comma-separated paths to dataset files (.jsonl) or dataset names")
    parser.add_argument("--base_model_path", type=str, default="/scratch/yl9038/models/Qwen3-14B",
                       help="Path to base model for answer generation")
    parser.add_argument("--judge_model_path", type=str, default="/scratch/yl9038/models/Qwen3-14B",
                       help="Path to judge model for consistency judgment")
    parser.add_argument("--output_path", type=str, default="./extracted_training_data.json",
                       help="Output path for training data")
    parser.add_argument("--max_examples_per_dataset", type=int, default=100,
                       help="Maximum number of examples to process per dataset")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="Use multiple GPUs for model loading")
    
    args = parser.parse_args()
    
    # Parse dataset paths
    dataset_paths = [p.strip() for p in args.dataset_paths.split(",")]
    
    logger.info(f"Processing {len(dataset_paths)} datasets:")
    for i, path in enumerate(dataset_paths, 1):
        logger.info(f"  {i}. {path}")
    
    process_multiple_datasets(
        dataset_paths,
        args.base_model_path,
        args.judge_model_path,
        args.output_path,
        args.max_examples_per_dataset,
        args.use_quantization,
        args.use_multi_gpu
    )

if __name__ == "__main__":
    main() 