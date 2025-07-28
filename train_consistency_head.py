#!/usr/bin/env python3
"""
Training script for adding and training a consistency head to Qwen3-32B model.
This script:
1. Loads Qwen3-32B model and adds a consistency output head
2. Generates trajectories on 100 HotpotQA training examples
3. Trains the consistency head to predict whether search answers are consistent with search results
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for XML-like tags
BEGIN_OF_SEARCH, END_OF_SEARCH = "<search>", "</search>"
BEGIN_OF_INFO, END_OF_INFO = "<information>", "</information>"
BEGIN_OF_ANSWER, END_OF_ANSWER = "<answer>", "</answer>"

@dataclass
class ConsistencyTrainingExample:
    """Data class for consistency training examples."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    consistency_label: float
    search_query: str
    search_results: str
    model_answer: str

class ConsistencyHead(nn.Module):
    """A simple consistency head that predicts whether search answer is consistent with search results."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the consistency head."""
        # Use the last hidden state for classification
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        last_hidden = self.dropout(last_hidden)
        consistency_score = self.classifier(last_hidden)  # [batch_size, 1]
        return consistency_score.squeeze(-1)  # [batch_size]

class ModelWithConsistencyHead(nn.Module):
    """Wrapper model that adds a consistency head to the base model."""
    
    def __init__(self, base_model: AutoModelForCausalLM, consistency_head: ConsistencyHead):
        super().__init__()
        self.base_model = base_model
        self.consistency_head = consistency_head
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                consistency_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with optional consistency prediction."""
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Use last layer hidden states
        
        # Predict consistency score
        consistency_score = self.consistency_head(hidden_states)
        
        outputs = {"consistency_score": consistency_score}
        
        if consistency_labels is not None:
            # Calculate loss
            loss = nn.BCELoss()(consistency_score, consistency_labels)
            outputs["loss"] = loss
        
        return outputs

class ConsistencyDataset(Dataset):
    """Dataset for consistency training."""
    
    def __init__(self, examples: List[ConsistencyTrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "consistency_label": example.consistency_label,
            "search_query": example.search_query,
            "search_results": example.search_results,
            "model_answer": example.model_answer
        }

def load_hotpotqa_data(data_path: str, num_examples: int = 100) -> List[Dict[str, Any]]:
    """Load HotpotQA training data."""
    logger.info(f"Loading HotpotQA data from {data_path}")
    
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    else:
        # Try loading as HuggingFace dataset
        dataset = load_dataset('json', data_files=data_path)
        data = dataset['train'].to_list()
    
    # Take first num_examples
    data = data[:num_examples]
    logger.info(f"Loaded {len(data)} examples from HotpotQA")
    return data

def extract_search_queries_and_answers(trajectory: str) -> List[Tuple[str, str, str]]:
    """
    Extract search queries, search results, and model answers from trajectory.
    Returns list of (search_query, search_results, model_answer) tuples.
    """
    search_blocks = []
    
    # Find all search blocks
    search_pattern = rf"{re.escape(BEGIN_OF_SEARCH)}(.*?){re.escape(END_OF_SEARCH)}"
    info_pattern = rf"{re.escape(BEGIN_OF_INFO)}(.*?){re.escape(END_OF_INFO)}"
    
    search_matches = list(re.finditer(search_pattern, trajectory, re.DOTALL))
    info_matches = list(re.finditer(info_pattern, trajectory, re.DOTALL))
    
    # Match search queries with their corresponding info blocks
    for i, search_match in enumerate(search_matches):
        search_query = search_match.group(1).strip()
        
        # Find the corresponding info block (should be right after the search)
        search_end = search_match.end()
        info_start = trajectory.find(BEGIN_OF_INFO, search_end)
        
        if info_start != -1:
            info_end = trajectory.find(END_OF_INFO, info_start)
            if info_end != -1:
                search_results = trajectory[info_start + len(BEGIN_OF_INFO):info_end].strip()
                
                # Find the model's answer after this search (look for <answer> tags)
                answer_start = trajectory.find(BEGIN_OF_ANSWER, info_end)
                if answer_start != -1:
                    answer_end = trajectory.find(END_OF_ANSWER, answer_start)
                    if answer_end != -1:
                        model_answer = trajectory[answer_start + len(BEGIN_OF_ANSWER):answer_end].strip()
                        search_blocks.append((search_query, search_results, model_answer))
    
    return search_blocks

def generate_trajectories(model, tokenizer, data: List[Dict[str, Any]], 
                         max_new_tokens: int = 1024) -> List[Dict[str, Any]]:
    """Generate trajectories using the model on the given data."""
    logger.info("Generating trajectories...")
    
    trajectories = []
    
    for item in tqdm(data, desc="Generating trajectories"):
        question = item["question"]
        
        # Create prompt for trajectory generation
        prompt = f"""Answer the following question step by step. When you need to search for information, use <search>query</search> tags. You will receive search results in <information>results</information> tags. Provide your final answer in <answer>answer</answer> tags.

Question: {question}

Let me think about this step by step:"""
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate trajectory
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated trajectory
        trajectory = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        generated_part = trajectory[len(prompt):]
        
        trajectories.append({
            "question": question,
            "trajectory": generated_part,
            "original_item": item
        })
    
    return trajectories

def judge_consistency(judge_model, judge_tokenizer, search_query: str, 
                     search_results: str, model_answer: str) -> float:
    """Use the judging model to determine consistency between search results and model answer."""
    
    prompt = f"""You are a judge that determines whether a model's answer to a search query is consistent with the search results.

Search Query: {search_query}

Search Results: {search_results}

Model's Answer: {model_answer}

Is the model's answer consistent with the search results? Answer with "Yes" or "No".

Answer:"""
    
    inputs = judge_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id
        )
    
    response = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip().lower()
    
    # Determine consistency score
    if "yes" in response:
        return 1.0
    elif "no" in response:
        return 0.0
    else:
        # If unclear, default to 0.5
        return 0.5

def create_training_examples(trajectories: List[Dict[str, Any]], 
                           judge_model, judge_tokenizer,
                           base_tokenizer) -> List[ConsistencyTrainingExample]:
    """Create training examples from trajectories."""
    logger.info("Creating training examples...")
    
    training_examples = []
    
    for trajectory_data in tqdm(trajectories, desc="Processing trajectories"):
        trajectory = trajectory_data["trajectory"]
        
        # Extract search blocks
        search_blocks = extract_search_queries_and_answers(trajectory)
        
        for search_query, search_results, model_answer in search_blocks:
            # Judge consistency
            consistency_score = judge_consistency(
                judge_model, judge_tokenizer, search_query, search_results, model_answer
            )
            
            # Create input for the model (context + search query + search results + model answer)
            input_text = f"""Search Query: {search_query}

Search Results: {search_results}

Model's Answer: {model_answer}

Consistency Score:"""
            
            # Tokenize input
            inputs = base_tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            )
            
            # Create training example
            example = ConsistencyTrainingExample(
                input_ids=inputs["input_ids"].squeeze(0),
                attention_mask=inputs["attention_mask"].squeeze(0),
                consistency_label=consistency_score,
                search_query=search_query,
                search_results=search_results,
                model_answer=model_answer
            )
            
            training_examples.append(example)
    
    logger.info(f"Created {len(training_examples)} training examples")
    return training_examples

def train_consistency_head(model: ModelWithConsistencyHead, 
                          train_dataset: ConsistencyDataset,
                          num_epochs: int = 5,
                          batch_size: int = 8,
                          learning_rate: float = 1e-4) -> None:
    """Train the consistency head."""
    logger.info("Training consistency head...")
    
    # Create data loader
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        consistency_labels = torch.tensor([item["consistency_label"] for item in batch], dtype=torch.float32)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "consistency_labels": consistency_labels
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Setup optimizer (only for consistency head parameters)
    optimizer = optim.AdamW(model.consistency_head.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            consistency_labels = batch["consistency_labels"].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                consistency_labels=consistency_labels
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train consistency head for Qwen3-32B")
    parser.add_argument("--model_path", type=str, default="/scratch/yl9038/models/Qwen3-32B",
                       help="Path to Qwen3-32B model")
    parser.add_argument("--data_path", type=str, default="./data/hotpotqa/train.jsonl",
                       help="Path to HotpotQA training data")
    parser.add_argument("--num_examples", type=int, default=100,
                       help="Number of examples to use for trajectory generation")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for consistency head")
    parser.add_argument("--output_dir", type=str, default="./trained_consistency_model",
                       help="Output directory for trained model")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization for memory efficiency")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model...")
    if args.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    # Create consistency head
    hidden_size = base_model.config.hidden_size
    consistency_head = ConsistencyHead(hidden_size)
    
    # Create model with consistency head
    model = ModelWithConsistencyHead(base_model, consistency_head)
    model.to(device)
    
    # Load HotpotQA data
    data = load_hotpotqa_data(args.data_path, args.num_examples)
    
    # Generate trajectories
    trajectories = generate_trajectories(base_model, tokenizer, data)
    
    # Create training examples (using the same model as judge for simplicity)
    training_examples = create_training_examples(trajectories, base_model, tokenizer, tokenizer)
    
    # Create dataset
    train_dataset = ConsistencyDataset(training_examples)
    
    # Train consistency head
    train_consistency_head(model, train_dataset, args.num_epochs, args.batch_size, args.learning_rate)
    
    # Save the trained model
    logger.info(f"Saving trained model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save consistency head
    torch.save(model.consistency_head.state_dict(), os.path.join(args.output_dir, "consistency_head.pt"))
    
    # Save model configuration
    config = {
        "hidden_size": hidden_size,
        "model_path": args.model_path,
        "training_args": vars(args)
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 