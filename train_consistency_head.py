#!/usr/bin/env python3
"""
Training script for adding and training a consistency head to Qwen3-32B model.
This script:
1. Loads Qwen3-32B model and adds a consistency output head
2. Loads pre-generated trajectories from trajectory_train_200.jsonl
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
    """A consistency head that predicts consistency at the </search> position."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Classifier that takes hidden states and predicts consistency
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the consistency head.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - hidden states for each token
            
        Returns:
            consistency_score: [batch_size] - consistency score at the </search> position
        """
        # Convert to float32 to avoid dtype mismatch with quantized models
        hidden_states = hidden_states.float()
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Take the last hidden state (at the </search> position)
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Predict consistency score for the </search> position
        # [batch_size, hidden_size] -> [batch_size, 1] -> [batch_size]
        consistency_score = self.classifier(last_hidden).squeeze(-1)
        
        return consistency_score

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
        """Forward pass with consistency prediction at </search> position."""
        # Get hidden states from base model
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Use last layer hidden states
        
        # Predict consistency score at the </search> position
        consistency_score = self.consistency_head(hidden_states)
        
        outputs = {"consistency_score": consistency_score}
        
        if consistency_labels is not None:
            # Calculate loss for consistency prediction
            loss_fct = nn.BCELoss()
            loss = loss_fct(consistency_score, consistency_labels)
            outputs["loss"] = loss
        
        return outputs
    


class ConsistencyDataset(Dataset):
    """Dataset for consistency training examples."""
    
    def __init__(self, examples: List[ConsistencyTrainingExample]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def save_training_examples(training_examples: List[ConsistencyTrainingExample], save_path: str) -> None:
    """Save processed training examples to disk."""
    logger.info(f"Saving {len(training_examples)} training examples to {save_path}")
    
    # Convert to serializable format
    data_to_save = []
    for example in training_examples:
        data_to_save.append({
            "input_ids": example.input_ids.tolist(),
            "attention_mask": example.attention_mask.tolist(),
            "consistency_label": example.consistency_label,
            "search_query": example.search_query,
            "search_results": example.search_results,
            "model_answer": example.model_answer
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=2)
    
    logger.info(f"Training examples saved successfully")

def load_training_examples(load_path: str) -> List[ConsistencyTrainingExample]:
    """Load processed training examples from disk."""
    logger.info(f"Loading training examples from {load_path}")
    
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_examples = []
    for item in data:
        example = ConsistencyTrainingExample(
            input_ids=torch.tensor(item["input_ids"], dtype=torch.long, device="cpu"),
            attention_mask=torch.tensor(item["attention_mask"], dtype=torch.long, device="cpu"),
            consistency_label=item["consistency_label"],
            search_query=item["search_query"],
            search_results=item["search_results"],
            model_answer=item["model_answer"]
        )
        training_examples.append(example)
    
    logger.info(f"Loaded {len(training_examples)} training examples")
    return training_examples

def load_trajectory_data(trajectory_paths: List[str]) -> List[Dict[str, Any]]:
    """Load pre-generated trajectories from multiple JSONL files."""
    logger.info(f"Loading trajectories from {len(trajectory_paths)} files")
    
    all_trajectories = []
    for trajectory_path in trajectory_paths:
        logger.info(f"Loading from {trajectory_path}")
        trajectories = []
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))
        
        logger.info(f"Loaded {len(trajectories)} trajectories from {trajectory_path}")
        all_trajectories.extend(trajectories)
    
    logger.info(f"Total trajectories loaded: {len(all_trajectories)}")
    return all_trajectories

def extract_search_blocks_and_answers(trajectory: str) -> List[Tuple[str, str, str, int]]:
    """Extract search blocks and their corresponding answers from trajectory."""
    search_blocks = []
    
    # Find all search blocks: <search>query</search><information>results</information>
    # Allow for whitespace between the tags
    search_pattern = r'<search>(.*?)</search>\s*<information>(.*?)</information>'
    
    # Use finditer to get both matches and their positions
    search_matches = list(re.finditer(search_pattern, trajectory, re.DOTALL))
    
    # Find the final answer
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, trajectory, re.DOTALL)
    final_answer = answer_match.group(1).strip() if answer_match else ""
    
    for match in search_matches:
        search_query = match.group(1).strip()
        search_results = match.group(2).strip()
        # Get the end position of the </search> tag
        search_end_pos = match.end(1) + len("</search>")
        search_blocks.append((search_query, search_results, final_answer, search_end_pos))
    
    return search_blocks

def generate_model_answer(model, tokenizer, search_query: str) -> str:
    """Generate an answer to the search query using the trained model WITHOUT search results."""
    # Create prompt for the model to answer the search query independently
    prompt = f"""Question: {search_query}

Answer:"""
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    answer = generated_text[len(prompt):].strip()
    
    return answer

def judge_consistency(judge_model, judge_tokenizer, search_query: str, 
                     search_results: str, model_answer: str) -> float:
    """Use referee model to judge if the central fact in model's answer aligns with search results."""
    
    # Create prompt for consistency judgment focusing on central facts
    prompt = f"""Search Query: {search_query}

Search Results: {search_results}

Model's Answer: {model_answer}

Question: Does the central fact or main information in the model's answer align with the search results? 
Focus only on the most important fact that answers the query. Ignore minor details, style differences, or additional information.
If the core fact is correct, answer "Yes". If the core fact contradicts or is missing, answer "No".

Answer:"""
    
    # Tokenize input
    inputs = judge_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
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
            do_sample=False,
            temperature=0.1,
            top_p=0.95,
            top_k=20,
            pad_token_id=judge_tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (after the prompt)
    judgment = generated_text[len(prompt):].strip().lower()
    
    # Determine consistency score based on central fact alignment
    if "yes" in judgment:
        return 1.0
    elif "no" in judgment:
        return 0.0
    else:
        # If unclear, default to 0.5 (uncertain)
        return 0.5

def create_training_examples(trajectories: List[Dict[str, Any]], 
                           trained_model, trained_tokenizer,
                           judge_model, judge_tokenizer) -> List[ConsistencyTrainingExample]:
    """Create training examples by extracting hidden states at </search> position.
    
    The process:
    1. Extract search query and results from trajectory
    2. Generate model answer using ONLY the search query (no search results)
    3. Judge consistency between the independent model answer and search results
    4. Use original context hidden states for training
    """
    training_examples = []
    
    logger.info(f"Processing {len(trajectories)} trajectories...")
    
    for i, trajectory_data in enumerate(tqdm(trajectories, desc="Processing trajectories")):
        trajectory = trajectory_data.get("trajectory", "")
        
        # Extract search blocks
        search_blocks = extract_search_blocks_and_answers(trajectory)
        
        for search_query, search_results, final_answer, search_end_pos in search_blocks:
            try:
                # Create the context up to </search> position
                # Use the position information from the regex match
                context_text = trajectory[:search_end_pos]
                
                # Tokenize the context
                context_inputs = trained_tokenizer(
                    context_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=False
                )
                
                # Move inputs to the same device as the model
                device = next(trained_model.parameters()).device
                context_inputs = {k: v.to(device) for k, v in context_inputs.items()}
                
                # Get hidden states at the </search> position
                with torch.no_grad():
                    outputs = trained_model(
                        input_ids=context_inputs["input_ids"],
                        attention_mask=context_inputs["attention_mask"],
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
                
                # Generate a new answer using the model for this search query (without search results)
                generated_answer = generate_model_answer(trained_model, trained_tokenizer, search_query)
                
                # Judge consistency between the generated answer and search results
                final_consistency = judge_consistency(judge_model, judge_tokenizer, search_query, search_results, generated_answer)
                
                # Create training example with the context up to </search>
                # Ensure tensors are on CPU to avoid device mismatch issues
                example = ConsistencyTrainingExample(
                    input_ids=context_inputs["input_ids"].squeeze(0).cpu(),
                    attention_mask=context_inputs["attention_mask"].squeeze(0).cpu(),
                    consistency_label=final_consistency,
                    search_query=search_query,
                    search_results=search_results,
                    model_answer=generated_answer
                )
                
                training_examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing search block {i}: {e}")
                import traceback
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue
    
    logger.info(f"Created {len(training_examples)} training examples")
    return training_examples

def train_consistency_head(model: ModelWithConsistencyHead, 
                          train_dataset: ConsistencyDataset,
                          device: torch.device,
                          num_epochs: int = 5,
                          batch_size: int = 8,
                          learning_rate: float = 1e-4) -> None:
    """Train the consistency head."""
    logger.info("Starting consistency head training...")
    
    # Create data loader
    def collate_fn(batch):
        # Pad sequences to the same length
        max_len = max(len(item.input_ids) for item in batch)
        
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            # Ensure tensors are on CPU for collation
            item_input_ids = item.input_ids.cpu() if item.input_ids.device.type == 'cuda' else item.input_ids
            item_attention_mask = item.attention_mask.cpu() if item.attention_mask.device.type == 'cuda' else item.attention_mask
            
            # Pad input_ids
            padding_length = max_len - len(item_input_ids)
            if padding_length > 0:
                padded_input_ids = torch.cat([
                    item_input_ids,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            else:
                padded_input_ids = item_input_ids
            input_ids.append(padded_input_ids)
            
            # Pad attention_mask
            padding_length = max_len - len(item_attention_mask)
            if padding_length > 0:
                padded_attention_mask = torch.cat([
                    item_attention_mask,
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            else:
                padded_attention_mask = item_attention_mask
            attention_masks.append(padded_attention_mask)
            
            labels.append(item.consistency_label)
        
        # Debug: Check tensor sizes and devices
        input_ids_sizes = [tensor.size(0) for tensor in input_ids]
        attention_mask_sizes = [tensor.size(0) for tensor in attention_masks]
        input_ids_devices = [tensor.device for tensor in input_ids]
        attention_mask_devices = [tensor.device for tensor in attention_masks]
        
        if len(set(input_ids_sizes)) > 1:
            logger.warning(f"Input IDs have different sizes: {input_ids_sizes}")
        if len(set(attention_mask_sizes)) > 1:
            logger.warning(f"Attention masks have different sizes: {attention_mask_sizes}")
        if len(set(input_ids_devices)) > 1:
            logger.warning(f"Input IDs have different devices: {input_ids_devices}")
        if len(set(attention_mask_devices)) > 1:
            logger.warning(f"Attention masks have different devices: {attention_mask_devices}")
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "consistency_labels": torch.tensor(labels, dtype=torch.float)
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Setup optimizer (only for consistency head parameters)
    optimizer = optim.AdamW(model.consistency_head.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            consistency_labels = batch["consistency_labels"].to(device)
            
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
    parser.add_argument("--trajectory_paths", nargs='+', 
                       default=["./data/hotpotqa/trajectory_train_200.jsonl", "./data/2wikimultihop/trajectory_train_200.jsonl"],
                       help="Paths to pre-generated trajectory data files")
    parser.add_argument("--processed_data_path", type=str, default="./processed_training_examples.json",
                       help="Path to save/load processed training examples")
    parser.add_argument("--force_reprocess", action="store_true",
                       help="Force reprocessing of training examples and rewrite processed_training_examples.json even if it exists")
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
    
    # Ensure base model is also on the correct device if not using device_map="auto"
    if not args.use_quantization:
        base_model.to(device)
    
    # Check if processed training examples already exist
    if os.path.exists(args.processed_data_path) and not args.force_reprocess:
        logger.info(f"Loading existing processed training examples from {args.processed_data_path}")
        training_examples = load_training_examples(args.processed_data_path)
        logger.info(f"Loaded {len(training_examples)} existing training examples")
    else:
        if args.force_reprocess and os.path.exists(args.processed_data_path):
            logger.info(f"Force reprocessing enabled. Will regenerate and overwrite {args.processed_data_path}")
        elif not os.path.exists(args.processed_data_path):
            logger.info(f"Processed data file {args.processed_data_path} not found. Generating new training examples.")
        logger.info("Generating labeled dataset using the consistency training method...")
        
        # Load pre-generated trajectories
        trajectories = load_trajectory_data(args.trajectory_paths)
        logger.info(f"Loaded {len(trajectories)} trajectories")
        
        # Create training examples by:
        # 1. For each </search> block, use trained model to generate answer
        # 2. Use referee model to judge consistency
        # 3. Set consistency_label = 1 if consistent, 0 if not
        # 4. Train consistency head at </search> position
        training_examples = create_training_examples(trajectories, base_model, tokenizer, base_model, tokenizer)
        
        if len(training_examples) == 0:
            logger.error("No training examples created! This means no search blocks were found in the trajectories.")
            logger.error("Please check that the trajectory files contain search blocks in the format: <search>query</search><information>results</information>")
            raise ValueError("No training examples could be created from the trajectories")
        
        logger.info(f"Successfully created {len(training_examples)} training examples")
        
        # Save processed training examples for future use
        save_training_examples(training_examples, args.processed_data_path)
    
    # Create dataset
    train_dataset = ConsistencyDataset(training_examples)
    
    # Train consistency head
    train_consistency_head(model, train_dataset, device, args.num_epochs, args.batch_size, args.learning_rate)
    
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