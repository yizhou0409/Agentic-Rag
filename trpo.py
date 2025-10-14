#!/usr/bin/env python3
"""
trpo.py - Online PPO Training on Trajectory Trees

Supports two modes:
1. Inference: Generate trajectory trees with F1 metrics (no training)
2. Online PPO: True online training - iteratively generate and train

Usage:
# Inference only (generate trees + compute F1)
python trpo.py --mode inference --max_data_samples 50 --output_dir output/trees

# Online PPO training
python trpo.py --mode online_ppo --ppo_iterations 5 --questions_per_iter 10 --output_dir output/ppo
"""

import argparse
import json
import logging
import os
import gc
from typing import List, Dict, Any, Tuple
from datetime import datetime
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from main import InferenceSystem, InferenceConfig
from utils import parse_reasoning_generation, calculate_metrics, extract_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TURN_WARNING = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."


# =============================================================================
# Utility Functions
# =============================================================================

def cleanup_cuda_memory():
    """Force cleanup of CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def load_dataset_questions(dataset_name: str, split: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load questions from dataset file."""
    dataset_path = f"data/{dataset_name}/{split}.jsonl"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    questions = []
    with open(dataset_path, 'r') as f:
        for line_num, line in enumerate(f):
            if max_samples and line_num >= max_samples:
                break
            data = json.loads(line.strip())
            
            if dataset_name == "hotpotqa":
                questions.append({
                    "id": data["id"],
                    "question": data["question"],
                    "golden_answers": data["golden_answers"]
                })
            elif dataset_name == "2wikimultihop":
                questions.append({
                    "id": data["_id"],
                    "question": data["question"],
                    "golden_answers": [data["answer"]]
                })
    
    logger.info(f"Loaded {len(questions)} questions from {dataset_name} {split}")
    return questions


def load_system(reasoner_model_name: str, summarizer_model_name: str, reasoner_lora_path: str, 
                retriever_type: str, retriever_index_path: str, e5_model_path: str) -> InferenceSystem:
    """Load inference system."""
    config = InferenceConfig(
        reasoner_model_name=reasoner_model_name,
        summarizer_model_name=summarizer_model_name,
        reasoner_lora_path=reasoner_lora_path,
        retriever_type=retriever_type,
        retriever_index_path=retriever_index_path,
        e5_model_path=e5_model_path
    )
    return InferenceSystem(config)


# =============================================================================
# Trajectory Tree Generation
# =============================================================================

def _count_nodes(node: Dict[str, Any]) -> int:
    """Count total nodes in the trajectory tree."""
    count = 1
    for child in node["children"]:
        count += _count_nodes(child)
    return count


def generate_trajectory_tree(question: Dict[str, Any], system: InferenceSystem, 
                            max_depth: int, max_first_width: int, max_width: int, 
                            max_branching_depth: int = None) -> Dict[str, Any]:
    """Generate a tree of trajectories for a given question."""
    if max_branching_depth is None:
        max_branching_depth = max_depth
    
    # Initialize root node
    root_node = {
        "id": question["id"],
        "question": question["question"],
        "golden_answers": question["golden_answers"],
        "sequence": "",
        "response": "",
        "turns": [],
        "final_turn": 0,
        "depth": 0,
        'parent': None,
        "children": [],
        "is_leaf": False,
        "answer": None,
        "search_query": None
    }
    
    # Apply chat template to initial sequence
    messages = [{"role": "user", "content": system.reasoner.prompt_template.format(question=question["question"])}]
    root_node["sequence"] = system.reasoner.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    
    # BFS traversal
    queue = [(root_node, 0)]
    leaf_nodes = []
    
    while queue:
        current_node, current_depth = queue.pop(0)
        queries = []
        
        if current_depth == max_depth - 1:
            current_node["sequence"] += MAX_TURN_WARNING
        
        # Determine number of branches
        if current_depth >= max_branching_depth:
            num_branches = 1
        elif current_depth == 0:
            num_branches = max_first_width
        else:
            num_branches = max_width
        
        # Perform retrieval if needed
        if current_node["search_query"]:
            retrieved_docs = system.retriever.search(current_node["search_query"])
            doc_texts = [doc.get('text', doc.get('contents', str(doc))) if isinstance(doc, dict) else str(doc) for doc in retrieved_docs]
            summary = system.summarizer.summarize_documents(current_node["search_query"], doc_texts)
            current_node["sequence"] += f"\n<information> {summary} </information>"
        
        # Generate branches
        generated_nodes = 0
        max_attempts = num_branches * 3
        attempt_count = 0
        
        while generated_nodes < num_branches and attempt_count < max_attempts:
            attempt_count += 1
            
            try:
                question_copy = {
                    "id": f"{question['id']}_branch_{current_depth}_{generated_nodes}",
                    "question": question["question"],
                    "golden_answers": question["golden_answers"],
                    "sequence": current_node["sequence"],
                    "response": "",
                    "turns": current_node["turns"].copy(),
                    "final_turn": len(current_node["turns"]),
                    "answer": None,
                    "error": None
                }
                inference_result = system.inference_one_turn([question_copy])
                updated_question = inference_result[0][0]
                search_query = inference_result[1].get(question_copy["id"], "")
                
                # Check for duplicate queries
                if search_query and search_query in queries:
                    continue
                
                generated_nodes += 1
                
                # Create child node
                child_node = {
                    "id": updated_question["id"],
                    "question": updated_question["question"],
                    "golden_answers": updated_question["golden_answers"],
                    "sequence": updated_question["sequence"],
                    "response": updated_question["response"],
                    "turns": updated_question["turns"],
                    "depth": current_depth + 1,
                    "parent": current_node,
                    "children": [],
                    "is_leaf": False,
                    "answer": updated_question.get("answer"),
                    "search_query": search_query
                }
                
                if not child_node["search_query"] or child_node["depth"] > max_depth:
                    child_node["is_leaf"] = True
                    leaf_nodes.append(child_node)
                else:
                    queue.append((child_node, current_depth + 1))
                    queries.append(search_query)
                
                current_node["children"].append(child_node)
                
            except Exception as e:
                logger.warning(f"Error generating branch: {e}")
                continue
    
    return {
        "question_id": question["id"],
        "question": question["question"],
        "golden_answers": question["golden_answers"],
        "root": root_node,
        "leaf_nodes": leaf_nodes,
        "max_depth": max_depth,
        "max_first_width": max_first_width,
        "max_width": max_width,
        "max_branching_depth": max_branching_depth,
        "total_nodes": _count_nodes(root_node)
    }


# =============================================================================
# Reward Computation
# =============================================================================

def compute_node_rewards(trajectory_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute rewards for all nodes in the trajectory tree.
    - Leaf nodes: scored by F1
    - Internal nodes: average of children's rewards
    """
    golden_answers = trajectory_tree["golden_answers"]
    
    def compute_rewards_recursive(node: Dict[str, Any]) -> float:
        if node["is_leaf"]:
            # Leaf node: compute F1 score as reward
            response_text = node.get("response", "")
            thought, parsed_search_query, parsed_answer = parse_reasoning_generation(response_text)
            final_answer = parsed_answer if parsed_answer else extract_answer(response_text)
            
            if golden_answers:
                metrics = calculate_metrics(final_answer, golden_answers)
                reward = metrics["f1"]
            else:
                reward = 0.0
            
            node["reward"] = reward
            return reward
        else:
            # Internal node: average of children's rewards
            if not node["children"]:
                node["reward"] = 0.0
                return 0.0
            
            child_rewards = [compute_rewards_recursive(child) for child in node["children"]]
            avg_reward = sum(child_rewards) / len(child_rewards)
            node["reward"] = avg_reward
            return avg_reward
    
    compute_rewards_recursive(trajectory_tree["root"])
    return trajectory_tree


def extract_training_data_from_tree(trajectory_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract training data from trajectory tree.
    Each sample contains sequence, response, reward, and advantage.
    """
    training_samples = []
    
    def extract_samples_recursive(node: Dict[str, Any]):
        # Skip root node (no search query to learn)
        if node["depth"] > 0 and node.get("search_query"):
            # Compute advantage relative to siblings
            if node.get("parent") and node["parent"]["children"]:
                siblings = node["parent"]["children"]
                sibling_rewards = [s["reward"] for s in siblings]
                avg_sibling_reward = sum(sibling_rewards) / len(sibling_rewards)
                advantage = node["reward"] - avg_sibling_reward
            else:
                advantage = 0.0
            
            # Create training sample
            sample = {
                "node_id": node["id"],
                "question_id": trajectory_tree["question_id"],
                "question": trajectory_tree["question"],
                "depth": node["depth"],
                "sequence": node.get("sequence", ""),
                "response": node.get("response", ""),
                "search_query": node["search_query"],
                "reward": node["reward"],
                "advantage": advantage,
                "is_leaf": node["is_leaf"]
            }
            training_samples.append(sample)
        
        # Recursively process children
        for child in node.get("children", []):
            extract_samples_recursive(child)
    
    extract_samples_recursive(trajectory_tree["root"])
    return training_samples


# =============================================================================
# PPO Training Components
# =============================================================================

class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory training samples with rewards.
    
    Label masking strategy:
    1. Context (previous turns): labels = -100 (no loss)
    2. Retrieved info blocks: labels = -100 (no loss)  
    3. Model-generated responses: labels = token_ids (compute loss)
    
    Example:
        Context: "Question: Who wrote Harry Potter?"  [-100, -100, ...]
        Response: "<think>I should search</think>"    [145, 4523, ...]  ← LOSS
                  "<search>Harry Potter author</search>" [678, 2341, ...]  ← LOSS
        
        Next turn context: "Question...<search>...</search>"  [-100, -100, ...]
                          "<information>Harry Potter is...</information>"  [-100, -100, ...]  ← MASKED!
        Next response: "<think>Now I know</think>"  [234, 567, ...]  ← LOSS
                      "<answer>J.K. Rowling</answer>"  [890, 123, ...]  ← LOSS
    """
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_length: int = 2048):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sequence = sample["sequence"]
        response = sample["response"]
        
        # Tokenize full sequence
        full_text = sequence + response
        full_tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize just the sequence (context)
        context_tokens = self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        context_length = context_tokens["input_ids"].shape[1]
        
        # Create labels: -100 for context tokens, actual tokens for response
        labels = full_tokens["input_ids"].clone()
        labels[:, :context_length] = -100
        
        # IMPORTANT: Mask out <information>...</information> blocks
        # These are retrieved from external sources, not generated by model
        # We should NOT compute loss on them
        if "<information>" in full_text:
            import re
            info_pattern = r'<information>.*?</information>'
            
            # Find all information blocks
            for match in re.finditer(info_pattern, full_text, re.DOTALL):
                info_start_char = match.start()
                info_end_char = match.end()
                
                # Map character positions to token positions
                # Decode tokens one by one to find boundaries
                current_text = ""
                start_token_idx = None
                end_token_idx = None
                
                for token_idx in range(full_tokens["input_ids"].shape[1]):
                    token_text = self.tokenizer.decode(full_tokens["input_ids"][0, :token_idx + 1], skip_special_tokens=False)
                    
                    # Check if we've reached the start of info block
                    if start_token_idx is None and len(token_text) >= info_start_char:
                        start_token_idx = token_idx
                    
                    # Check if we've reached the end of info block
                    if start_token_idx is not None and len(token_text) >= info_end_char:
                        end_token_idx = token_idx + 1
                        break
                
                # Mask out these tokens
                if start_token_idx is not None and end_token_idx is not None:
                    labels[:, start_token_idx:end_token_idx] = -100
        
        return {
            "input_ids": full_tokens["input_ids"].squeeze(0),
            "attention_mask": full_tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "advantage": torch.tensor(sample["advantage"], dtype=torch.float32)
        }


def compute_ppo_loss(
    model, ref_model, input_ids, attention_mask, labels, advantages,
    clip_ratio: float = 0.2, kl_coef: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute PPO loss with ratio clipping and KL divergence penalty.
    
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Normalize advantages (handle small batches where std is unreliable)
    adv_std = advantages.std()
    if advantages.numel() <= 1 or adv_std < 1e-8:
        # Just center, don't normalize (std is unreliable with 1 sample or constant values)
        advantages_normalized = advantages - advantages.mean()
    else:
        advantages_normalized = (advantages - advantages.mean()) / (adv_std + 1e-8)
    
    # Forward pass on current model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    
    # Forward pass on reference model (no gradients)
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        ref_logits = ref_outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    
    # Get log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
    
    # Mask for valid tokens
    mask = (shift_labels != -100).float()
    
    batch_size, seq_len, vocab_size = shift_logits.shape
    shift_labels_masked = shift_labels.clone()
    shift_labels_masked[shift_labels == -100] = 0
    
    # Gather log probs for target tokens
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels_masked.unsqueeze(-1)).squeeze(-1)
    token_ref_log_probs = torch.gather(ref_log_probs, dim=-1, index=shift_labels_masked.unsqueeze(-1)).squeeze(-1)
    
    # Mask out padding
    token_log_probs = token_log_probs * mask
    token_ref_log_probs = token_ref_log_probs * mask
    
    # Compute per-sample log probs
    sample_log_probs = token_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    sample_ref_log_probs = token_ref_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    
    # Importance sampling ratio: π_new / π_old
    ratio = torch.exp(sample_log_probs - sample_ref_log_probs)
    
    # PPO clipped objective
    surrogate1 = ratio * advantages_normalized
    surrogate2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages_normalized
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
    # KL divergence penalty
    kl_div = F.kl_div(
        log_probs.view(-1, vocab_size),
        ref_log_probs.view(-1, vocab_size).exp(),
        reduction='none',
        log_target=False
    ).sum(dim=-1).view(batch_size, seq_len)
    kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-8)
    
    # Total loss
    total_loss = policy_loss + kl_coef * kl_div
    
    # Metrics
    metrics = {
        'policy_loss': policy_loss.item(),
        'kl_div': kl_div.item(),
        'total_loss': total_loss.item(),
        'mean_ratio': ratio.mean().item(),
        'ratio_clipped_frac': ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float().mean().item()
    }
    
    return total_loss, metrics


def train_one_iteration(
    model, ref_model, tokenizer, training_samples: List[Dict[str, Any]],
    num_epochs: int, batch_size: int, learning_rate: float, warmup_steps: int,
    clip_ratio: float, kl_coef: float, logging_steps: int
) -> Dict[str, Any]:
    """Train for one PPO iteration."""
    
    dataset = TrajectoryDataset(training_samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    model.train()
    stats = {"policy_losses": [], "kl_divs": [], "mean_ratios": [], "ratio_clipped_fracs": []}
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            advantages = batch["advantage"].to(model.device)
            rewards = batch["reward"]
            
            loss, metrics = compute_ppo_loss(
                model, ref_model, input_ids, attention_mask, labels,
                advantages, clip_ratio, kl_coef
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Log stats
            stats["policy_losses"].append(metrics['policy_loss'])
            stats["kl_divs"].append(metrics['kl_div'])
            stats["mean_ratios"].append(metrics['mean_ratio'])
            stats["ratio_clipped_fracs"].append(metrics['ratio_clipped_frac'])
            
            # Update progress bar with metrics
            progress_bar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'policy': f"{metrics['policy_loss']:.4f}",
                'kl': f"{metrics['kl_div']:.4f}",
                'ratio': f"{metrics['mean_ratio']:.3f}",
                'clip%': f"{metrics['ratio_clipped_frac']*100:.1f}",
                'reward': f"{rewards.mean().item():.3f}",
                'adv': f"{advantages.mean().item():.3f}"
            })
    
    return stats


# =============================================================================
# Main Functions
# =============================================================================

def main_inference(args):
    """Inference mode: Generate trees, compute F1, save JSONs (no training)."""
    logger.info("=" * 80)
    logger.info("MODE: INFERENCE ONLY")
    logger.info("=" * 80)
    
    # Load all questions
    all_questions = load_dataset_questions(args.dataset_name, args.split, max_samples=None)
    
    # Determine question range
    start_idx = args.start_sample - 1 if args.start_sample else 0  # Convert to 0-indexed
    if args.end_sample:
        end_idx = args.end_sample  # Inclusive in user terms, exclusive in Python slicing
    else:
        end_idx = start_idx + args.max_data_samples if args.max_data_samples else len(all_questions)
    
    # Ensure within bounds
    end_idx = min(end_idx, len(all_questions))
    questions = all_questions[start_idx:end_idx]
    
    logger.info(f"Processing questions {start_idx + 1} to {end_idx} (total: {len(questions)})")
    
    system = load_system(args.reasoner_model_name, args.summarizer_model_name, args.reasoner_lora_path,
                        args.retriever_type, args.retriever_index_path, args.e5_model_path)
    
    system.reasoner.model.eval()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    trees_dir = os.path.join(args.output_dir, "trees")
    os.makedirs(trees_dir, exist_ok=True)
    
    trajectory_trees = []
    start_time = time.time()
    
    for question_idx, question in enumerate(tqdm(questions, desc="Generating trees")):
        try:
            # Generate tree
            trajectory_tree = generate_trajectory_tree(
                question, system, args.max_depth, args.max_first_width,
                args.max_width, args.max_branching_depth
            )
            
            # Compute rewards
            trajectory_tree = compute_node_rewards(trajectory_tree)
            trajectory_trees.append(trajectory_tree)
            
            # Save tree as JSON
            tree_json = {
                "question_id": trajectory_tree["question_id"],
                "question": trajectory_tree["question"],
                "golden_answers": trajectory_tree["golden_answers"],
                "root_reward": trajectory_tree["root"].get("reward", 0.0),
                "total_nodes": trajectory_tree["total_nodes"],
                "leaf_count": len(trajectory_tree["leaf_nodes"])
            }
            tree_path = os.path.join(trees_dir, f"question_{question_idx+1:03d}_{question['id']}_tree.json")
            with open(tree_path, 'w') as f:
                json.dump(tree_json, f, indent=2)
            
            cleanup_cuda_memory()
        except Exception as e:
            logger.error(f"Error processing question {question['id']}: {e}")
            continue
    
    generation_time = time.time() - start_time
    
    # Compute F1 metrics
    best_f1_scores = []
    average_f1_scores = []
    for tree in trajectory_trees:
        leaf_f1_scores = [leaf.get('reward', 0.0) for leaf in tree.get('leaf_nodes', [])]
        if leaf_f1_scores:
            best_f1_scores.append(max(leaf_f1_scores))
            average_f1_scores.append(sum(leaf_f1_scores) / len(leaf_f1_scores))
    
    # Save summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "mode": "inference",
        "total_questions": len(trajectory_trees),
        "generation_time_seconds": generation_time,
        "generation_time_hours": generation_time / 3600,
        "f1_metrics": {
            "overall_best_f1": max(best_f1_scores) if best_f1_scores else 0.0,
            "average_best_f1": sum(best_f1_scores) / len(best_f1_scores) if best_f1_scores else 0.0,
            "overall_average_f1": sum(average_f1_scores) / len(average_f1_scores) if average_f1_scores else 0.0,
            "questions_with_perfect_f1": sum(1 for f1 in best_f1_scores if f1 >= 1.0),
            "questions_with_zero_f1": sum(1 for f1 in best_f1_scores if f1 == 0.0)
        },
        "trajectory_stats": {
            "avg_nodes_per_tree": sum(t['total_nodes'] for t in trajectory_trees) / len(trajectory_trees) if trajectory_trees else 0,
            "avg_leaf_nodes": sum(len(t['leaf_nodes']) for t in trajectory_trees) / len(trajectory_trees) if trajectory_trees else 0,
            "total_trajectories": sum(len(t['leaf_nodes']) for t in trajectory_trees)
        }
    }
    
    summary_path = os.path.join(args.output_dir, "inference_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("INFERENCE COMPLETE!")
    logger.info(f"  Questions: {len(trajectory_trees)}")
    logger.info(f"  Overall Best F1: {summary['f1_metrics']['overall_best_f1']:.4f}")
    logger.info(f"  Average Best F1: {summary['f1_metrics']['average_best_f1']:.4f}")
    logger.info(f"  Time: {generation_time:.2f}s ({generation_time/3600:.2f}h)")
    logger.info("=" * 80)


def main_online_ppo(args):
    """Online PPO mode: Iteratively generate trees and train."""
    logger.info("=" * 80)
    logger.info("MODE: ONLINE PPO TRAINING")
    logger.info("=" * 80)
    logger.info(f"PPO Iterations: {args.ppo_iterations}")
    logger.info(f"Questions per iteration: {args.questions_per_iter}")
    logger.info(f"Epochs per iteration: {args.rl_num_epochs}")
    
    # Load all questions
    all_questions = load_dataset_questions(args.dataset_name, args.split, max_samples=None)
    logger.info(f"Loaded {len(all_questions)} total questions")
    
    # Determine starting index
    start_idx = args.start_sample - 1 if args.start_sample else 0  # Convert to 0-indexed
    logger.info(f"Starting from question index: {start_idx} (question #{start_idx + 1})")
    
    # Determine ending index
    if args.end_sample:
        end_idx = args.end_sample  # end_sample is inclusive, but Python slicing is exclusive
        total_questions_to_process = end_idx - start_idx
    else:
        total_questions_to_process = args.ppo_iterations * args.questions_per_iter
        end_idx = start_idx + total_questions_to_process
    
    # Ensure we don't exceed dataset size
    end_idx = min(end_idx, len(all_questions))
    total_questions_to_process = end_idx - start_idx
    
    logger.info(f"Processing questions {start_idx + 1} to {end_idx} (total: {total_questions_to_process})")
    
    # Load system
    system = load_system(args.reasoner_model_name, args.summarizer_model_name, args.reasoner_lora_path,
                        args.retriever_type, args.retriever_index_path, args.e5_model_path)
    
    model = system.reasoner.model
    tokenizer = system.reasoner.tokenizer
    
    # Load reference model (frozen base model)
    logger.info(f"Loading reference model (frozen): {args.reasoner_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.reasoner_model_name, torch_dtype="auto", device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Reference model loaded and frozen")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    iterations_dir = os.path.join(args.output_dir, "iterations")
    os.makedirs(iterations_dir, exist_ok=True)
    
    total_start_time = time.time()
    iteration_stats = []
    
    # PPO iterations
    for iteration in range(args.ppo_iterations):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"PPO ITERATION {iteration + 1}/{args.ppo_iterations}")
        logger.info("=" * 80)
        
        iteration_start_time = time.time()
        
        # Periodic reference model refresh (prevents clipping ceiling)
        if args.ref_refresh_interval > 0 and iteration > 0 and iteration % args.ref_refresh_interval == 0:
            logger.info(f"🔄 Refreshing reference model at iteration {iteration + 1}")
            ref_model.load_state_dict(model.state_dict())
            ref_model.eval()
            logger.info("Reference model synchronized with current policy")
        
        # Get questions for this iteration (sequential, not random)
        iter_start_idx = start_idx + (iteration * args.questions_per_iter)
        iter_end_idx = min(iter_start_idx + args.questions_per_iter, end_idx)
        
        # Check if we've processed all questions
        if iter_start_idx >= end_idx:
            logger.warning(f"Reached end of question range at iteration {iteration + 1}")
            break
        
        questions_batch = all_questions[iter_start_idx:iter_end_idx]
        logger.info(f"Processing questions {iter_start_idx + 1} to {iter_end_idx} ({len(questions_batch)} questions)")
        
        # Generate trajectory trees with CURRENT policy
        logger.info("Generating trees with current policy...")
        trajectory_trees = []
        all_training_samples = []
        
        model.eval()  # Eval mode for tree generation
        
        for question in tqdm(questions_batch, desc=f"Iter {iteration+1} - Gen"):
            try:
                tree = generate_trajectory_tree(
                    question, system, args.max_depth, args.max_first_width,
                    args.max_width, args.max_branching_depth
                )
                tree = compute_node_rewards(tree)
                samples = extract_training_data_from_tree(tree)
                
                all_training_samples.extend(samples)
                trajectory_trees.append(tree)
                cleanup_cuda_memory()
            except Exception as e:
                logger.warning(f"Error: {e}")
                continue
        
        logger.info(f"Generated {len(trajectory_trees)} trees, {len(all_training_samples)} samples")
        
        if len(all_training_samples) == 0:
            logger.warning(f"No samples in iteration {iteration + 1}, skipping...")
            continue
        
        # Train on generated trees
        logger.info(f"Training on {len(all_training_samples)} samples...")
        iter_dir = os.path.join(iterations_dir, f"iteration_{iteration + 1}")
        os.makedirs(iter_dir, exist_ok=True)
        
        training_stats = train_one_iteration(
            model, ref_model, tokenizer, all_training_samples,
            args.rl_num_epochs, args.rl_batch_size, args.rl_learning_rate,
            args.rl_warmup_steps, args.rl_clip_ratio, args.rl_kl_coef,
            args.rl_logging_steps
        )
        
        # Save model after this iteration
        model.save_pretrained(os.path.join(iter_dir, "model"))
        tokenizer.save_pretrained(os.path.join(iter_dir, "model"))
        
        iteration_time = time.time() - iteration_start_time
        
        # Compute iteration stats
        best_f1_scores = []
        for tree in trajectory_trees:
            leaf_f1s = [leaf.get('reward', 0.0) for leaf in tree.get('leaf_nodes', [])]
            if leaf_f1s:
                best_f1_scores.append(max(leaf_f1s))
        
        iter_stat = {
            "iteration": iteration + 1,
            "num_questions": len(trajectory_trees),
            "num_samples": len(all_training_samples),
            "avg_best_f1": sum(best_f1_scores) / len(best_f1_scores) if best_f1_scores else 0,
            "iteration_time": iteration_time,
            "final_policy_loss": training_stats["policy_losses"][-1] if training_stats["policy_losses"] else 0,
            "final_kl_div": training_stats["kl_divs"][-1] if training_stats["kl_divs"] else 0,
            "avg_kl_div": sum(training_stats["kl_divs"]) / len(training_stats["kl_divs"]) if training_stats["kl_divs"] else 0
        }
        iteration_stats.append(iter_stat)
        
        logger.info(f"Iteration {iteration + 1} complete:")
        logger.info(f"  Trees: {len(trajectory_trees)}, Samples: {len(all_training_samples)}")
        logger.info(f"  Avg best F1: {iter_stat['avg_best_f1']:.4f}")
        logger.info(f"  Final KL div: {iter_stat['final_kl_div']:.6f}")
        logger.info(f"  Time: {iteration_time:.2f}s")
    
    total_time = time.time() - total_start_time
    
    # Save final summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "mode": "online_ppo",
        "ppo_iterations": args.ppo_iterations,
        "questions_per_iteration": args.questions_per_iter,
        "total_time_seconds": total_time,
        "total_time_hours": total_time / 3600,
        "iteration_stats": iteration_stats
    }
    
    summary_path = os.path.join(args.output_dir, "online_ppo_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 80)
    logger.info("ONLINE PPO COMPLETE!")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
    logger.info(f"Final model: {iterations_dir}/iteration_{args.ppo_iterations}/model/")
    logger.info("=" * 80)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Tree Generation and Online PPO Training")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["inference", "online_ppo"], required=True,
                       help="Mode: 'inference' (generate trees + F1), 'online_ppo' (iterative training)")
    
    # Common parameters
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (hotpotqa, 2wikimultihop)")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split")
    parser.add_argument("--reasoner_model_name", type=str, required=True, help="Base model name")
    parser.add_argument("--summarizer_model_name", type=str, required=True, help="Summarizer model")
    parser.add_argument("--reasoner_lora_path", type=str, default="", help="LoRA adapter path")
    parser.add_argument("--retriever_type", type=str, required=True, help="Retriever type")
    parser.add_argument("--retriever_index_path", type=str, required=True, help="Retriever index")
    parser.add_argument("--e5_model_path", type=str, required=True, help="E5 model path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # Tree parameters
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum tree depth")
    parser.add_argument("--max_first_width", type=int, default=3, help="Branches at first level")
    parser.add_argument("--max_width", type=int, default=2, help="Branches at other levels")
    parser.add_argument("--max_branching_depth", type=int, default=None, help="Stop branching after this depth")
    
    # Inference mode parameters
    parser.add_argument("--max_data_samples", type=int, default=50, help="Max questions for inference mode (if start_sample/end_sample not specified)")
    
    # Common parameters for both modes
    parser.add_argument("--start_sample", type=int, default=None, help="Starting question number (1-indexed, inclusive). Default: 1")
    parser.add_argument("--end_sample", type=int, default=None, help="Ending question number (1-indexed, inclusive). Default: auto-calculated based on mode")
    
    # Online PPO mode parameters
    parser.add_argument("--ppo_iterations", type=int, default=5, help="Number of PPO iterations")
    parser.add_argument("--questions_per_iter", type=int, default=10, help="Questions per iteration (processed sequentially)")
    parser.add_argument("--rl_num_epochs", type=int, default=2, help="Epochs per iteration")
    parser.add_argument("--rl_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--rl_warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--rl_clip_ratio", type=float, default=0.2, help="PPO clipping ratio")
    parser.add_argument("--rl_kl_coef", type=float, default=0.1, help="KL divergence coefficient")
    parser.add_argument("--rl_logging_steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--ref_refresh_interval", type=int, default=10, 
                        help="Refresh reference model every N iterations (0 = never refresh)")
    
    args = parser.parse_args()
    
    if args.mode == "inference":
        main_inference(args)
    elif args.mode == "online_ppo":
        main_online_ppo(args)

