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
    """Consistency head that predicts consistency score with multiple layers."""
    
    def __init__(self, hidden_size: int, layer_sizes: List[int] = None, target_layer: int = -1):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [hidden_size, 512, 256, 1]
        
        self.target_layer = target_layer
        self.layers = nn.ModuleList()
        
        # Build layers
        for i in range(len(layer_sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.ReLU() if i < len(layer_sizes) - 2 else nn.Identity(),  # No activation on final layer
                nn.Dropout(0.1) if i < len(layer_sizes) - 2 else nn.Identity()  # No dropout on final layer
            )
            self.layers.append(layer)
    
    def forward(self, hidden_states, target_positions=None):
        # If target_positions is provided, use those specific positions
        # Otherwise, fall back to the last token (for backward compatibility)
        if target_positions is not None:
            # Extract hidden states at specific target positions
            target_layer_hs = hidden_states[self.target_layer]
            batch_size = target_layer_hs.shape[0]
            
            # Ensure we're using the same device and dtype as the hidden states
            target_hidden = torch.zeros(batch_size, target_layer_hs.shape[-1], 
                                      device=target_layer_hs.device, 
                                      dtype=target_layer_hs.dtype)
            
            for i, pos in enumerate(target_positions):
                if pos < target_layer_hs.shape[1]:
                    target_hidden[i] = target_layer_hs[i, pos, :]
                else:
                    # Fallback to last position if target position is out of bounds
                    target_hidden[i] = target_layer_hs[i, -1, :]
        else:
            # Take the last token's hidden state from the target layer (original behavior)
            target_hidden = hidden_states[self.target_layer][:, -1, :]
        
        # Pass through all layers
        x = target_hidden
        for layer in self.layers:
            x = layer(x)
        
        # Apply sigmoid to get probabilities (0-1 range) for MSE loss
        consistency_score = torch.sigmoid(x)
        return consistency_score

class ModelWithConsistencyHead(nn.Module):
    """Wrapper model that adds consistency head to base model."""
    
    def __init__(self, base_model: AutoModelForCausalLM, consistency_head: ConsistencyHead):
        super().__init__()
        self.base_model = base_model
        self.consistency_head = consistency_head
    
    def forward(self, **kwargs):
        # Extract target_positions if provided
        target_positions = kwargs.pop('target_positions', None)
        
        # Get base model outputs
        outputs = self.base_model(**kwargs, output_hidden_states=True)
        
        # Convert all hidden states to float32 and ensure they're on the same device
        # Get the device of the consistency head
        consistency_head_device = next(self.consistency_head.parameters()).device
        
        hidden_states = []
        for hs in outputs.hidden_states:
            # Convert to float32 and move to consistency head device
            # Handle the case where hidden states might be on different devices
            if hs.device != consistency_head_device:
                hs_float32 = hs.to(dtype=torch.float32, device=consistency_head_device)
            else:
                hs_float32 = hs.to(dtype=torch.float32)
            hidden_states.append(hs_float32)
        
        # Predict consistency score with target positions
        consistency_score = self.consistency_head(hidden_states, target_positions)
        
        return {
            "logits": outputs.logits,
            "consistency_score": consistency_score
        }

def load_model(model_path: str, use_quantization: bool = False, use_multi_gpu: bool = True):
    """Load model without quantization on 2 GPUs using device_map."""
    
    logger.info(f"Loading model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {available_gpus}")
    
    if available_gpus < 2:
        logger.warning(f"Only {available_gpus} GPUs available, but 2 GPUs requested. Using available GPUs.")
        use_multi_gpu = False
    
    # Configure device map for multi-GPU without quantization
    if use_multi_gpu and available_gpus >= 2:
        # Use device_map to distribute model across 2 GPUs
        # H20 GPUs have 80GB memory each, so we can use most of it
        device_map = "auto"
        max_memory = {0: "75GB", 1: "75GB", "cpu": "100GB"}
        logger.info("Using device_map='auto' for 2 H20 GPUs without quantization (75GB each)")
    else:
        device_map = "auto"
        max_memory = None
        logger.info("Using single GPU configuration")
    
    # Load base model without quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    logger.info("Model loaded successfully!")
    return model, tokenizer

def ensure_same_device(model: ModelWithConsistencyHead, device: torch.device):
    """Ensure all model components are on the same device."""
    logger.info(f"Moving all model components to device: {device}")
    
    # For device_map models, we need to be more careful about moving
    # The base model might be distributed across multiple devices
    # We'll move the consistency head to the target device and ensure
    # the forward pass handles device placement properly
    
    # Move consistency head to device
    model.consistency_head = model.consistency_head.to(device)
    
    # Ensure consistency head parameters are on the target device
    for name, param in model.consistency_head.named_parameters():
        if param.device != device:
            logger.warning(f"Consistency head parameter {name} is on {param.device}, moving to {device}")
            param.data = param.data.to(device)
    
    logger.info("Consistency head moved to target device")
    logger.info("Note: Base model remains distributed across GPUs for memory efficiency")

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
    
    # Find the position of the '>' token at the end of '</search>'
    target_position = find_closing_search_position(input_text, tokenizer)
    
    return inputs, target_position

def find_closing_search_position(text: str, tokenizer: AutoTokenizer) -> int:
    """Find the position of the '>' token at the end of '</search>' in the tokenized sequence."""
    
    # Find the last occurrence of '</search>' in the text
    search_end = text.rfind('</search>')
    if search_end == -1:
        # If no '</search>' found, return the last position
        logger.warning("No '</search>' found in text, using last position")
        return -1
    
    # Find the position of the '>' character
    closing_bracket_pos = search_end + len('</search>') - 1  # Position of '>'
    
    # Tokenize the text up to the closing bracket position
    text_before_closing = text[:closing_bracket_pos + 1]
    tokens = tokenizer.encode(text_before_closing, add_special_tokens=False)
    
    # The target position is the last token (which should be the '>' token)
    target_position = len(tokens) - 1
    
    # Debug: Log the target token
    if target_position >= 0:
        target_token = tokenizer.decode([tokens[target_position]]) if target_position < len(tokens) else "OUT_OF_BOUNDS"
        logger.debug(f"Target position {target_position}, token: '{target_token}'")
    
    return target_position

def train_consistency_head(model: ModelWithConsistencyHead, tokenizer: AutoTokenizer, 
                          training_data: List[Dict[str, Any]], 
                          num_epochs: int = 10, batch_size: int = 4,
                          learning_rate: float = 3e-4, save_path: str = "./trained_consistency_head.pt",
                          validation_split: float = 0.2, early_stopping_patience: int = 3,
                          scheduler_type: str = "step", step_size: int = 50, gamma: float = 0.9):
    """Train the consistency head with proper handling of DataParallel models."""
    
    logger.info("Starting consistency head training...")
    
    # Ensure consistency head is on the target device
    # Use GPU 0 as the primary device for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensure_same_device(model, device)
    
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
    
    # Learning rate scheduler - choose between step-based and plateau-based
    if scheduler_type == "step":
        # Step-based scheduler that reduces LR every N steps
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
        logger.info(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
    else:
        # Plateau-based scheduler that reduces LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        logger.info("Using ReduceLROnPlateau scheduler")
    
    # Loss function for consistency prediction - MSE for regression
    consistency_loss_fn = nn.MSELoss()
    
    # Training loop with early stopping
    model.train()
    best_val_accuracy = 0.0
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
                    inputs, target_position = create_training_input(question, search_context, tokenizer)
                    
                    # Move to device (use the device already defined)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    consistency_label = torch.tensor([[consistency_label]], dtype=torch.float32).to(device)
                    
                    # Add target position to inputs
                    inputs['target_positions'] = [target_position]
                    
                    # Forward pass
                    outputs = model(**inputs)
                    predicted_score = outputs['consistency_score']
                    
                    # Ensure predicted_score is on the correct device
                    if predicted_score.device != device:
                        predicted_score = predicted_score.to(device)
                    
                    # Debug: Check for NaN in predictions
                    if torch.isnan(predicted_score).any():
                        logger.warning(f"NaN detected in predictions")
                        continue
                    
                    # Calculate loss (MSE loss)
                    loss = consistency_loss_fn(predicted_score, consistency_label)
                    
                    # Debug: Check for NaN in loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected, predicted: {predicted_score.item():.6f}, target: {consistency_label.item():.6f}")
                        continue
                    
                    batch_loss += loss
                    
                    # Count correct predictions (threshold at 0.5)
                    pred_binary = (predicted_score > 0.5).float()
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
                    inputs, target_position = create_training_input(question, search_context, tokenizer)
                    
                    # Move to device (use the device already defined)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    consistency_label = torch.tensor([[consistency_label]], dtype=torch.float32).to(device)
                    
                    # Add target position to inputs
                    inputs['target_positions'] = [target_position]
                    
                    # Forward pass
                    outputs = model(**inputs)
                    predicted_score = outputs['consistency_score']
                    
                    # Ensure predicted_score is on the correct device
                    if predicted_score.device != device:
                        predicted_score = predicted_score.to(device)
                    
                    # Calculate loss (MSE loss)
                    loss = consistency_loss_fn(predicted_score, consistency_label)
                    val_loss += loss.item()
                    
                    # Count correct predictions
                    pred_binary = (predicted_score > 0.5).float()
                    target_binary = (consistency_label > 0.5).float()
                    val_correct += (pred_binary == target_binary).sum().item()
                    val_total += 1
                    
                except Exception as e:
                    logger.error(f"Error processing validation example: {e}")
                    continue
        
        avg_val_loss = val_loss / len(val_data) if val_data else 0
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        
        # Update learning rate scheduler
        if scheduler_type == "step":
            scheduler.step()  # StepLR doesn't need validation loss
        else:
            scheduler.step(avg_val_loss)  # ReduceLROnPlateau needs validation loss
        
        logger.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
                   f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}, "
                   f"LR = {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
            
            # Save consistency head state dict, ensuring it's on CPU for saving
            consistency_head_state = model.consistency_head.state_dict()
            # Move all tensors to CPU for saving
            cpu_state = {k: v.cpu() for k, v in consistency_head_state.items()}
            torch.save(cpu_state, save_path)
        else:
            patience_counter += 1
            logger.info(f"Validation accuracy didn't improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered!")
                break
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Perform detailed validation on the best model
    logger.info("Performing detailed validation on best model...")
    
    # Load the best model
    model.consistency_head.load_state_dict(torch.load(save_path))
    
    # Perform detailed validation
    validation_report = detailed_validation(model, tokenizer, val_data, device)
    
    # Save detailed validation report
    report_path, summary_path = save_validation_report(validation_report, save_path)
    
    # Print summary to console
    metrics = validation_report['overall_metrics']
    analysis = validation_report['detailed_analysis']
    
    logger.info("=" * 60)
    logger.info("FINAL VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Examples: {metrics['total_examples']}")
    logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
    logger.info(f"Wrong Predictions: {metrics['wrong_predictions']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"Average Loss: {metrics['avg_loss']:.6f}")
    logger.info("")
    logger.info(f"False Positives: {len(analysis['false_positives'])}")
    logger.info(f"False Negatives: {len(analysis['false_negatives'])}")
    logger.info(f"High Confidence Errors: {len(analysis['high_confidence_errors'])}")
    logger.info(f"Low Confidence Errors: {len(analysis['low_confidence_errors'])}")
    logger.info("=" * 60)
    logger.info(f"Detailed report saved to: {report_path}")
    logger.info(f"Summary saved to: {summary_path}")

def detailed_validation(model: ModelWithConsistencyHead, tokenizer: AutoTokenizer, 
                       val_data: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    """Perform detailed validation and return all wrong classifications."""
    
    logger.info("Performing detailed validation...")
    
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    # Store detailed results
    wrong_classifications = []
    correct_classifications = []
    
    with torch.no_grad():
        for idx, example in enumerate(val_data):
            try:
                # Get validation data
                question = example['question']
                search_context = example['search_context']
                consistency_label = example['consistency_label']
                
                # Create input
                inputs, target_position = create_training_input(question, search_context, tokenizer)
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                consistency_label = torch.tensor([[consistency_label]], dtype=torch.float32).to(device)
                
                # Add target position to inputs
                inputs['target_positions'] = [target_position]
                
                # Forward pass
                outputs = model(**inputs)
                predicted_score = outputs['consistency_score']
                
                # Ensure predicted_score is on the correct device
                if predicted_score.device != device:
                    predicted_score = predicted_score.to(device)
                
                # Calculate loss (MSE loss)
                loss = nn.MSELoss()(predicted_score, consistency_label)
                val_loss += loss.item()
                
                # Count correct predictions
                pred_binary = (predicted_score > 0.5).float()
                target_binary = (consistency_label > 0.5).float()
                is_correct = (pred_binary == target_binary).sum().item()
                val_correct += is_correct
                val_total += 1
                
                # Store detailed results
                result = {
                    'example_id': idx,
                    'question': question,
                    'search_query': example.get('search_query', 'N/A'),
                    'ground_truth': consistency_label.item(),
                    'predicted_score': predicted_score.item(),
                    'predicted_binary': pred_binary.item(),
                    'target_binary': target_binary.item(),
                    'is_correct': bool(is_correct),
                    'loss': loss.item(),
                    'target_position': target_position
                }
                
                if is_correct:
                    correct_classifications.append(result)
                else:
                    wrong_classifications.append(result)
                    
            except Exception as e:
                logger.error(f"Error processing validation example {idx}: {e}")
                continue
    
    avg_val_loss = val_loss / len(val_data) if val_data else 0
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    
    # Create detailed validation report
    validation_report = {
        'overall_metrics': {
            'total_examples': val_total,
            'correct_predictions': val_correct,
            'wrong_predictions': len(wrong_classifications),
            'accuracy': val_accuracy,
            'avg_loss': avg_val_loss
        },
        'wrong_classifications': wrong_classifications,
        'correct_classifications': correct_classifications,
        'detailed_analysis': {
            'false_positives': [r for r in wrong_classifications if r['predicted_binary'] == 1 and r['target_binary'] == 0],
            'false_negatives': [r for r in wrong_classifications if r['predicted_binary'] == 0 and r['target_binary'] == 1],
            'high_confidence_errors': [r for r in wrong_classifications if abs(r['predicted_score'] - 0.5) > 0.3],
            'low_confidence_errors': [r for r in wrong_classifications if abs(r['predicted_score'] - 0.5) <= 0.3]
        }
    }
    
    return validation_report

def save_validation_report(validation_report: Dict[str, Any], save_path: str):
    """Save detailed validation report to file."""
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save full report as JSON
    report_path = save_path.replace('.pt', '_validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    # Create human-readable summary
    summary_path = save_path.replace('.pt', '_validation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CONSISTENCY HEAD VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall metrics
        metrics = validation_report['overall_metrics']
        f.write(f"OVERALL METRICS:\n")
        f.write(f"  Total Examples: {metrics['total_examples']}\n")
        f.write(f"  Correct Predictions: {metrics['correct_predictions']}\n")
        f.write(f"  Wrong Predictions: {metrics['wrong_predictions']}\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Average Loss: {metrics['avg_loss']:.6f}\n\n")
        
        # Detailed analysis
        analysis = validation_report['detailed_analysis']
        f.write(f"DETAILED ANALYSIS:\n")
        f.write(f"  False Positives (predicted high consistency, actually low): {len(analysis['false_positives'])}\n")
        f.write(f"  False Negatives (predicted low consistency, actually high): {len(analysis['false_negatives'])}\n")
        f.write(f"  High Confidence Errors (|pred - 0.5| > 0.3): {len(analysis['high_confidence_errors'])}\n")
        f.write(f"  Low Confidence Errors (|pred - 0.5| <= 0.3): {len(analysis['low_confidence_errors'])}\n\n")
        
        # Wrong classifications details
        f.write(f"WRONG CLASSIFICATIONS ({len(validation_report['wrong_classifications'])} total):\n")
        f.write("=" * 80 + "\n")
        
        for i, wrong_case in enumerate(validation_report['wrong_classifications']):
            f.write(f"\n{i+1}. Example ID: {wrong_case['example_id']}\n")
            f.write(f"   Ground Truth: {wrong_case['ground_truth']:.4f} (binary: {wrong_case['target_binary']})\n")
            f.write(f"   Predicted: {wrong_case['predicted_score']:.4f} (binary: {wrong_case['predicted_binary']})\n")
            f.write(f"   Loss: {wrong_case['loss']:.6f}\n")
            f.write(f"   Target Position: {wrong_case['target_position']}\n")
            f.write(f"   Question: {wrong_case['question'][:200]}{'...' if len(wrong_case['question']) > 200 else ''}\n")
            f.write(f"   Search Query: {wrong_case['search_query'][:200]}{'...' if len(wrong_case['search_query']) > 200 else ''}\n")
            f.write("-" * 80 + "\n")
        
        # Correct classifications summary
        f.write(f"\nCORRECT CLASSIFICATIONS ({len(validation_report['correct_classifications'])} total):\n")
        f.write("=" * 80 + "\n")
        
        for i, correct_case in enumerate(validation_report['correct_classifications'][:10]):  # Show first 10
            f.write(f"\n{i+1}. Example ID: {correct_case['example_id']}\n")
            f.write(f"   Ground Truth: {correct_case['ground_truth']:.4f} (binary: {correct_case['target_binary']})\n")
            f.write(f"   Predicted: {correct_case['predicted_score']:.4f} (binary: {correct_case['predicted_binary']})\n")
            f.write(f"   Loss: {correct_case['loss']:.6f}\n")
        
        if len(validation_report['correct_classifications']) > 10:
            f.write(f"\n... and {len(validation_report['correct_classifications']) - 10} more correct classifications.\n")
    
    logger.info(f"Validation report saved to: {report_path}")
    logger.info(f"Validation summary saved to: {summary_path}")
    
    return report_path, summary_path

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
    parser.add_argument("--use_quantization", action="store_true", default=False,
                       help="Use 4-bit quantization (default: False)")
    parser.add_argument("--use_multi_gpu", action="store_true", default=True,
                       help="Use multiple GPUs for model loading (default: True)")
    parser.add_argument("--validation_split", type=float, default=0.2,
                       help="Fraction of data to use for validation")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Number of epochs to wait before early stopping")
    parser.add_argument("--layer_sizes", type=str, default="512,256",
                       help="Comma-separated list of layer sizes for consistency head (excluding input and output)")
    parser.add_argument("--target_layer", type=int, default=-1,
                       help="Which layer to probe (-1 for last layer, 0 for first layer, etc.)")
    parser.add_argument("--scheduler_type", type=str, default="step", choices=["step", "plateau"],
                       help="Type of learning rate scheduler: 'step' or 'plateau'")
    parser.add_argument("--step_size", type=int, default=50,
                       help="Step size for StepLR scheduler (every N steps)")
    parser.add_argument("--gamma", type=float, default=0.9,
                       help="Gamma (decay factor) for StepLR scheduler")
    
    args = parser.parse_args()
    
    # Load base model
    base_model, tokenizer = load_model(args.model_path, args.use_quantization, args.use_multi_gpu)
    
    # Get hidden size from base model
    hidden_size = base_model.config.hidden_size
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Parse layer sizes
    layer_sizes = [hidden_size] + [int(size.strip()) for size in args.layer_sizes.split(',')] + [1]
    logger.info(f"Consistency head layer sizes: {layer_sizes}")
    
    # Log target layer information
    if args.target_layer == -1:
        logger.info(f"Probing the last layer (layer {args.target_layer})")
    else:
        logger.info(f"Probing layer {args.target_layer}")
    
    # Create consistency head with float32 for numerical stability
    consistency_head = ConsistencyHead(hidden_size, layer_sizes=layer_sizes, target_layer=args.target_layer).to(dtype=torch.float32)
    
    # Create model with consistency head
    model = ModelWithConsistencyHead(base_model, consistency_head)
    
    # Device placement will be handled in train_consistency_head function
    # to ensure all tensors are on the same device
    
    # Count parameters in consistency head
    consistency_head_params = sum(p.numel() for p in model.consistency_head.parameters())
    logger.info(f"Consistency head parameters: {consistency_head_params:,}")
    
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
        args.validation_split, args.early_stopping_patience,
        args.scheduler_type, args.step_size, args.gamma
    )

if __name__ == "__main__":
    main() 