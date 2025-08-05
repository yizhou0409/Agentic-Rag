#!/usr/bin/env python3
"""
Training script for consistency head on 32B model.
This script:
1. Loads 32B model and adds consistency head
2. Trains only the consistency head
3. Uses question + search_context to generate next word
4. Trains consistency head output to match consistency_label
5. Saves the trained consistency head
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsistencyHead(nn.Module):
    """Consistency head that predicts consistency score."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]
        # Predict consistency score (0-1 range)
        consistency_score = torch.sigmoid(self.linear(last_hidden))
        return consistency_score

class ModelWithConsistencyHead(nn.Module):
    """Wrapper model that adds consistency head to base model."""
    
    def __init__(self, base_model: AutoModelForCausalLM, consistency_head: ConsistencyHead):
        super().__init__()
        self.base_model = base_model
        self.consistency_head = consistency_head
    
    def forward(self, **kwargs):
        # Get base model outputs
        outputs = self.base_model(**kwargs, output_hidden_states=True)
        
        # Get hidden states from the last layer and convert to float32
        hidden_states = outputs.hidden_states[-1].to(dtype=torch.float32)
        
        # Predict consistency score
        consistency_score = self.consistency_head(hidden_states)
        
        return {
            "logits": outputs.logits,
            "consistency_score": consistency_score
        }

def load_model(model_path: str, use_quantization: bool = True, use_multi_gpu: bool = False):
    """Load model with the same approach as extract_training_data.py."""
    
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

def prepare_training_data(training_data_path: str) -> List[Dict[str, Any]]:
    """Load and prepare training data."""
    logger.info(f"Loading training data from {training_data_path}")
    
    with open(training_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training examples")
    return data

def create_training_input(question: str, search_context: str, tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """Create input for training with question + search_context."""
    
    # Create input text: question + search_context
    input_text = f"{question}\n{search_context}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False
    )
    
    return inputs

def train_consistency_head(model: ModelWithConsistencyHead, tokenizer: AutoTokenizer, 
                          training_data: List[Dict[str, Any]], 
                          num_epochs: int = 10, batch_size: int = 4,
                          learning_rate: float = 3e-4, save_path: str = "./trained_consistency_head.pt",
                          validation_split: float = 0.2, early_stopping_patience: int = 3):
    """Train the consistency head with improved hyperparameters for the dataset."""
    
    logger.info("Starting consistency head training...")
    
    # Split data into train and validation
    import random
    random.seed(42)
    random.shuffle(training_data)
    
    split_idx = int(len(training_data) * (1 - validation_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    logger.info(f"Training examples: {len(train_data)}, Validation examples: {len(val_data)}")
    
    # Set up optimizer (only for consistency head)
    optimizer = optim.AdamW(model.consistency_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Loss function for consistency prediction - MSE for regression
    consistency_loss_fn = nn.MSELoss()
    
    # Training loop with early stopping
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Process training data in batches
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_loss = 0
            
            for example in batch_data:
                try:
                    # Get training data
                    question = example['question']
                    search_context = example['search_context']
                    consistency_label = example['consistency_label']
                    
                    # Create input
                    inputs = create_training_input(question, search_context, tokenizer)
                    
                    # Move to device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    consistency_label = torch.tensor([[consistency_label]], dtype=torch.float32).to(device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    predicted_consistency = outputs['consistency_score']
                    
                    # Debug: Check for NaN in predictions
                    if torch.isnan(predicted_consistency).any():
                        logger.warning(f"NaN detected in predictions")
                        continue
                    
                    # Calculate loss
                    loss = consistency_loss_fn(predicted_consistency, consistency_label)
                    
                    # Debug: Check for NaN in loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected, predicted: {predicted_consistency.item():.6f}, target: {consistency_label.item():.6f}")
                        continue
                    
                    batch_loss += loss
                    
                    # Count correct predictions (threshold at 0.5)
                    pred_binary = (predicted_consistency > 0.5).float()
                    target_binary = (consistency_label > 0.5).float()
                    train_correct += (pred_binary == target_binary).sum().item()
                    train_total += 1
                    
                except Exception as e:
                    logger.error(f"Error processing training example: {e}")
                    continue
            
            # Backward pass for batch
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.consistency_head.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += batch_loss.item()
        
        avg_train_loss = train_loss / (len(train_data) // batch_size + 1)
        train_accuracy = train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for example in val_data:
                try:
                    # Get validation data
                    question = example['question']
                    search_context = example['search_context']
                    consistency_label = example['consistency_label']
                    
                    # Create input
                    inputs = create_training_input(question, search_context, tokenizer)
                    
                    # Move to device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    consistency_label = torch.tensor([[consistency_label]], dtype=torch.float32).to(device)
                    
                    # Forward pass
                    outputs = model(**inputs)
                    predicted_consistency = outputs['consistency_score']
                    
                    # Calculate loss
                    loss = consistency_loss_fn(predicted_consistency, consistency_label)
                    val_loss += loss.item()
                    
                    # Count correct predictions
                    pred_binary = (predicted_consistency > 0.5).float()
                    target_binary = (consistency_label > 0.5).float()
                    val_correct += (pred_binary == target_binary).sum().item()
                    val_total += 1
                    
                except Exception as e:
                    logger.error(f"Error processing validation example: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_data) if val_data else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                   f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}, "
                   f"LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            torch.save(model.consistency_head.state_dict(), save_path)
        else:
            patience_counter += 1
            logger.info(f"Validation loss didn't improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered!")
                break
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train consistency head on 32B model")
    parser.add_argument("--model_path", type=str, default="/scratch/yl9038/models/Qwen3-32B",
                       help="Path to base 32B model")
    parser.add_argument("--training_data_path", type=str, default="./extracted_training_data.json",
                       help="Path to extracted training data")
    parser.add_argument("--save_path", type=str, default="./trained_consistency_head.pt",
                       help="Path to save trained consistency head")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate for consistency head")
    parser.add_argument("--use_quantization", action="store_true",
                       help="Use 4-bit quantization")
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="Use multiple GPUs for model loading")
    parser.add_argument("--validation_split", type=float, default=0.2,
                       help="Fraction of data to use for validation")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Number of epochs to wait before early stopping")
    
    args = parser.parse_args()
    
    # Load base model
    base_model, tokenizer = load_model(args.model_path, args.use_quantization, args.use_multi_gpu)
    
    # Get hidden size from base model
    hidden_size = base_model.config.hidden_size
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Create consistency head with float32 for numerical stability
    consistency_head = ConsistencyHead(hidden_size).to(dtype=torch.float32)
    
    # Create model with consistency head
    model = ModelWithConsistencyHead(base_model, consistency_head)
    
    # Move consistency head to the same device as base model
    device = next(base_model.parameters()).device
    model.consistency_head = model.consistency_head.to(device)
    
    # Freeze base model parameters (only train consistency head)
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    logger.info("Base model parameters frozen, only consistency head will be trained")
    
    # Load training data
    training_data = prepare_training_data(args.training_data_path)
    
    # Train consistency head
    train_consistency_head(
        model, tokenizer, training_data,
        args.num_epochs, args.batch_size,
        args.learning_rate, args.save_path,
        args.validation_split, args.early_stopping_patience
    )

if __name__ == "__main__":
    main() 