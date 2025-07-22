#!/usr/bin/env python3
"""
Modified train_supervised.py that follows LlamaFactory patterns
but doesn't require LlamaFactory installation.
Focuses on learning from trajectories without caring about answer correctness.
"""

import os
import json
import torch
import argparse
import logging
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    set_seed
)
from typing import List, Dict, Any

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

class LlamaFactoryStyleDataset(Dataset):
    """
    Dataset class that follows LlamaFactory patterns for trajectory learning.
    Focuses on learning the reasoning process without caring about final answer correctness.
    """
    
    def __init__(self, data_paths: List[str], tokenizer, max_length=2048, template="default"):
        """
        Args:
            data_paths: List of paths to trajectory data files
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            template: Template style (default, alpaca, etc.)
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        
        for data_path in data_paths:
            logger.info(f"Loading data from {data_path}")
            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            sample = self._process_item(item)
                            if sample:
                                self.samples.append(sample)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at line {line_num} in {data_path}: {e}")
                            continue
            logger.info(f"Loaded {len([s for s in self.samples if s])} samples from {data_path}")
        
        logger.info(f"Total samples loaded: {len(self.samples)}")

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single item into training format."""
        question = item.get("question", "")
        trajectory = item.get("trajectory", "")
        
        if not question or not trajectory:
            return None
        
        # Format according to template
        if self.template == "default":
            # Simple format: question + trajectory
            full_text = f"Question: {question}\n\nAnswer: {trajectory}"
        elif self.template == "alpaca":
            # Alpaca format
            full_text = f"### Instruction:\n{question}\n\n### Response:\n{trajectory}"
        else:
            # Default format
            full_text = f"Question: {question}\n\nAnswer: {trajectory}"
        
        # Tokenize with padding and truncation
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Pad to max_length
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM training)
        labels = enc.input_ids.clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Train model on trajectory data (LlamaFactory style)")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="/scratch/yl9038/models/Qwen3-8B-Base",
                       help="Path to the pretrained model")
    parser.add_argument("--template", type=str, default="default",
                       choices=["default", "alpaca"],
                       help="Template style for formatting data")
    
    # Data arguments
    parser.add_argument("--data_paths", type=str, nargs="+", required=True,
                       help="Paths to trajectory data files")
    parser.add_argument("--dataset_dir", type=str, default="data",
                       help="Dataset directory")
    parser.add_argument("--max_samples", type=int, default=100000,
                       help="Maximum number of samples to use")
    parser.add_argument("--val_size", type=float, default=0.15,
                       help="Validation set size ratio")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output/sft_checkpoints",
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=float, default=10.0,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=50,
                       help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--logging_steps", type=int, default=5,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluation steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                       help="Maximum number of checkpoints to save")
    parser.add_argument("--cutoff_len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--compute_type", type=str, default="pure_bf16",
                       choices=["fp16", "bf16", "pure_bf16"],
                       help="Compute type")
    parser.add_argument("--report_to", type=str, default="none",
                       help="Report to (none, wandb, tensorboard)")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--ddp_timeout", type=int, default=180000000,
                       help="DDP timeout")
    parser.add_argument("--dataloader_num_workers", type=int, default=1,
                       help="Number of dataloader workers")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info(f"Training arguments: {args}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if "bf16" in args.compute_type else torch.float16
    )
    
    # Load dataset
    dataset = LlamaFactoryStyleDataset(
        args.data_paths,
        tokenizer,
        max_length=args.cutoff_len,
        template=args.template
    )
    
    # Limit samples if specified
    if args.max_samples and len(dataset) > args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]
        logger.info(f"Limited dataset to {args.max_samples} samples")
    
    # Split into train and eval
    train_size = int((1 - args.val_size) * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    logger.info(f"Train size: {train_size}, Eval size: {eval_size}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,  # Disable this to avoid strategy mismatch
        fp16=args.compute_type == "fp16",
        bf16="bf16" in args.compute_type,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_timeout=args.ddp_timeout,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Initialize trainer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train and save
    logger.info("Starting training")
    trainer.train()
    
    logger.info(f"Saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 