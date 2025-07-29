#!/usr/bin/env python3
"""
Inference script for the trained consistency head.
This script demonstrates how to use the trained consistency head to predict
consistency scores for new search query-answer pairs.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Any, Optional
import argparse

from train_consistency_head import ConsistencyHead, ModelWithConsistencyHead

def load_trained_model(model_path: str, consistency_head_path: str, 
                      use_quantization: bool = True):
    """Load the base model and add consistency head manually."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Get hidden size from base model
    hidden_size = base_model.config.hidden_size
    
    # Create and load consistency head
    consistency_head = ConsistencyHead(hidden_size)
    
    # Load consistency head weights if they exist
    if os.path.exists(consistency_head_path):
        print(f"Loading consistency head from {consistency_head_path}")
        consistency_head.load_state_dict(torch.load(consistency_head_path, map_location="cpu"))
    else:
        print(f"Warning: Consistency head file {consistency_head_path} not found. Using untrained head.")
    
    # Create model with consistency head
    model = ModelWithConsistencyHead(base_model, consistency_head)
    
    # Move consistency head to the same device as base model
    device = next(base_model.parameters()).device
    model.consistency_head = model.consistency_head.to(device)
    
    return model, tokenizer

def predict_consistency_at_search_position(model: ModelWithConsistencyHead, tokenizer, 
                                         search_query: str, search_results: str) -> float:
    """Predict consistency score at the </search> position."""
    
    # Create input context up to </search> position
    context_text = f"""<search>{search_query}</search>
<information>{search_results}</information>"""
    
    # Tokenize context
    inputs = tokenizer(
        context_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        padding=False
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict consistency at </search> position
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        consistency_score = outputs["consistency_score"].item()
    
    return consistency_score

def predict_consistency(model: ModelWithConsistencyHead, tokenizer, 
                       search_query: str, search_results: str, model_answer: str) -> float:
    """Predict consistency score for a given search query, results, and answer."""
    
    # Create input text
    input_text = f"""Search Query: {search_query}

Search Results: {search_results}

Model's Answer: {model_answer}

Consistency Score:"""
    
    # Tokenize input
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048,
        padding=True
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict consistency
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        consistency_scores = outputs["consistency_scores"]
        # Take the final consistency score
        consistency_score = consistency_scores[:, -1].item()
    
    return consistency_score

def main():
    parser = argparse.ArgumentParser(description="Inference with trained consistency head")
    parser.add_argument("--model_path", type=str, default="/scratch/yl9038/models/Qwen3-32B",
                       help="Path to base Qwen3-32B model")
    parser.add_argument("--consistency_head_path", type=str, default="./trained_consistency_model/consistency_head.pt",
                       help="Path to trained consistency head")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading trained model...")
    model, tokenizer = load_trained_model(
        args.model_path, 
        args.consistency_head_path, 
        args.use_quantization
    )
    
    # Example predictions at </search> position
    examples = [
        {
            "search_query": "What is the capital of France?",
            "search_results": "Paris is the capital and largest city of France.",
            "expected": "High consistency score (should be close to 1.0)"
        },
        {
            "search_query": "What is the capital of France?",
            "search_results": "Paris is the capital and largest city of France.",
            "expected": "High consistency score (should be close to 1.0)"
        },
        {
            "search_query": "Who wrote Romeo and Juliet?",
            "search_results": "William Shakespeare wrote Romeo and Juliet, a tragedy about two young lovers.",
            "expected": "High consistency score (should be close to 1.0)"
        }
    ]
    
    print("\n" + "="*80)
    print("CONSISTENCY PREDICTION AT </search> POSITION")
    print("="*80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Search Query: {example['search_query']}")
        print(f"Search Results: {example['search_results']}")
        print(f"Expected: {example['expected']}")
        print("-" * 60)
        
        # Predict consistency at </search> position
        consistency_score = predict_consistency_at_search_position(
            model, tokenizer,
            example["search_query"],
            example["search_results"]
        )
        
        print(f"Consistency Score at </search> position: {consistency_score:.4f}")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your own examples (press Ctrl+C to exit):")
    
    try:
        while True:
            print("\n" + "-" * 40)
            try:
                search_query = input("Search Query: ").strip()
                if not search_query:
                    continue
                    
                search_results = input("Search Results: ").strip()
                if not search_results:
                    continue
                    
                model_answer = input("Model Answer: ").strip()
                if not model_answer:
                    continue
                
                consistency_score = predict_consistency(
                    model, tokenizer, search_query, search_results, model_answer
                )
                
                print(f"\nPredicted Consistency Score: {consistency_score:.4f}")
                if consistency_score > 0.7:
                    print("Interpretation: HIGH consistency")
                elif consistency_score > 0.3:
                    print("Interpretation: MEDIUM consistency")
                else:
                    print("Interpretation: LOW consistency")
                    
            except EOFError:
                print("\n\nNo input available. Exiting interactive mode...")
                break
                
    except KeyboardInterrupt:
        print("\n\nExiting...")

if __name__ == "__main__":
    main() 