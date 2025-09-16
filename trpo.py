#!/usr/bin/env python3
"""
train_grpo.py

GRPO (Group Relative Policy Optimization) training script that:
- uses Reasoner from main.py to sample trajectories (so generation matches inference)
- masks tokens inside <information>...</information> in continuations when computing log-probs
- loads all generated trajectories for each question and trains via GRPO objective
- uses group-based ranking where all trajectories for a question form a group

Usage example:
python train_grpo.py \
  --data grpo_training_data.json \
  --base-model /scratch/yl9038/models/Qwen3-0.6B \
  --lora-adapter models/llamafactory/qwen3-lora-0.6B \
  --output-dir grpo_out \
  --device cuda \
  --epochs 1

"""

import argparse
import json
import logging
import os
import re
import time
import gc
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

def log_cuda_memory(stage: str = ""):
    """Log detailed CUDA memory usage information"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        
        logger.info(f"CUDA Memory {stage}:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB") 
        logger.info(f"  Max Allocated: {max_allocated:.2f} GB")
        logger.info(f"  Total Memory: {total_memory:.2f} GB")
        logger.info(f"  Free: {total_memory - reserved:.2f} GB")
        
        # Get detailed memory info
        memory_stats = torch.cuda.memory_stats(device)
        logger.info(f"  Active Allocations: {memory_stats.get('num_alloc_retries', 'N/A')}")
        logger.info(f"  OOM Count: {memory_stats.get('num_ooms', 'N/A')}")
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'total_memory': total_memory,
            'free': total_memory - reserved
        }
    else:
        logger.info(f"CUDA not available {stage}")
        return None

def cleanup_cuda_memory():
    """Force cleanup of CUDA memory"""
    if torch.cuda.is_available():
        logger.info("Cleaning up CUDA memory...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        log_cuda_memory("after cleanup")

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import PeftModel

# Import Reasoner and InferenceConfig from your main.py
# main.py must be next to this script and must not execute its main() on import (your main.py uses if __name__ == "__main__":)
from main import  InferenceSystem, InferenceConfig
from utils import is_exact_match, load_dataset, parse_reasoning_generation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# -----------------------------
# Answer extraction / normalization
# -----------------------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

MAX_TURN_WARNING = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."

def initialize_questions(items: List[dict]) -> List[Dict[str, Any]]:
    questions = []
    id = 0
    # initialize questions
    for it in items:
        # Extract required fields
        qid = it.get("id")
        if not qid:
            qid = f"trpo_{id}"
            id += 1
        prompt = it.get("instruction")
        input_text = it.get("input")
        ref_traj = it.get("reference_trajectory") or it.get("ref") or it.get("reference") or it.get("output")
        if not qid:
            logger.warning(f"Skipping item missing id: {it}")
            continue
        if not prompt:
            logger.warning(f"Skipping item {qid} missing prompt: {it}")
            continue
        if not input_text:
            logger.warning(f"Skipping item {qid} missing input: {it}")
            continue
        if not ref_traj:
            logger.warning(f"Item {qid} missing reference_trajectory; still kept (ref empty).")
            ref_traj = ""
        
        # Build conversation from prompt and input (similar to SFT format)
        if input_text:
            # Use input as the main question, prompt as context
            full_question = f"{prompt}\n\n{input_text}"
        else:
            # Fallback to just prompt if no input
            full_question = prompt
        
        

        questions.append({
            "id": qid,
            "question": full_question,
            "reference_trajectory": ref_traj,
            "golden_answers": [parse_reasoning_generation(ref_traj)[2]] if ref_traj else []
        })
    return questions

def load_system(reasoner_model_name: str, summarizer_model_name: str, reasoner_lora_path: Optional[str], retriever_type: str, retriever_index_path: str, e5_model_path: str) -> InferenceSystem:
    config = InferenceConfig(
        reasoner_model_name=reasoner_model_name,
        summarizer_model_name=summarizer_model_name,
        reasoner_lora_path=reasoner_lora_path,
        retriever_type=retriever_type,
        retriever_index_path=retriever_index_path,
        e5_model_path=e5_model_path
    )
    return InferenceSystem(config)

def _count_nodes(node: Dict[str, Any]) -> int:
    """Count total nodes in the trajectory tree."""
    count = 1  # Count current node
    for child in node["children"]:
        count += _count_nodes(child)
    return count

def generate_trajectory_tree(question: Dict[str, Any], system: InferenceSystem, max_depth: int, max_first_width: int, max_width: int) -> Dict[str, Any]:
    """
    Generate a tree of trajectories for a given question using the InferenceSystem.
    
    Args:
        question: Question data with 'id', 'question', 'golden_answers'
        system: InferenceSystem instance for generating responses
        max_depth: Maximum depth of the trajectory tree
        max_first_width: Maximum number of branches at the first level
        max_width: Maximum number of branches at subsequent levels
        
    Returns:
        Dictionary containing the trajectory tree structure
    """
    # Initialize the root node
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    
    # Debug: Log the initial sequence
    logger.info(f"Initial sequence for question {question['id']}: {root_node['sequence'][:300]}...")
    
    # Queue for BFS traversal: (node, parent_depth)
    queue = [(root_node, 0)]
    leaf_nodes = []
    
    while queue:
        current_node, current_depth = queue.pop(0)
        queries = []
        if current_depth == max_depth - 1:
            current_node["sequence"] += MAX_TURN_WARNING
        # Generate multiple trajectories from current node

        # Determine how many branches to generate
        if current_depth == 0:
            num_branches = max_first_width
        else:
            num_branches = max_width
            
        # Create multiple copies of the current question for parallel inference
        generated_nodes = 0

        if current_node["search_query"]:
            # Perform retrieval and add to response
            retrieved_docs = system.retriever.search(current_node["search_query"])
            doc_texts = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    doc_text = doc.get('text', doc.get('contents', str(doc)))
                else:
                    doc_text = str(doc)
                doc_texts.append(doc_text)
            # Summarize the retrieved documents
            summary = system.summarizer.summarize_documents(current_node["search_query"],doc_texts)
            # Add summarized information to the sequence
            info_text = f"\n<information> {summary} </information>"
            current_node["sequence"] += info_text

        # Add loop protection to prevent infinite loops
        max_attempts = num_branches * 3  # Allow up to 3x attempts to account for duplicates
        attempt_count = 0
        
        while generated_nodes < num_branches and attempt_count < max_attempts:
            attempt_count += 1
            logger.debug(f"Attempt {attempt_count}/{max_attempts} to generate branch {generated_nodes + 1}/{num_branches}")
            
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

                logger.info(f"Search query: {search_query}")
                
                # Check for duplicate search queries - THIS IS THE KEY FIX
                if search_query and search_query in queries:
                    logger.info(f"Duplicate search query '{search_query}' detected, skipping this attempt")
                    continue  # Skip this attempt but don't increment generated_nodes
                else:
                # Successfully generated a unique branch
                    generated_nodes += 1
                        # Create the child node
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
                logger.warning(f"Error generating branch {generated_nodes} for question {question['id']}: {e}")
                continue
        
        # Check if we hit the attempt limit
        if attempt_count >= max_attempts:
            logger.warning(f"Reached maximum attempts ({max_attempts}) for generating branches at depth {current_depth}, get {generated_nodes} branches")

    
    # Return the complete trajectory tree
    return {
        "question_id": question["id"],
        "question": question["question"],
        "golden_answers": question["golden_answers"],
        "root": root_node,
        "leaf_nodes": leaf_nodes,
        "max_depth": max_depth,
        "max_first_width": max_first_width,
        "max_width": max_width,
        "total_nodes": _count_nodes(root_node)
    }


def save_trajectory_tree_readable(trajectory_tree: Dict[str, Any], output_path: str):
    """
    Save trajectory tree in a human-readable format.
    
    Args:
        trajectory_tree: The trajectory tree structure
        output_path: Path to save the readable output
    """
    def format_node(node: Dict[str, Any], depth: int = 0) -> str:
        indent = "  " * depth
        lines = []
        
        lines.append(f"{indent}Node ID: {node['id']}")
        lines.append(f"{indent}Depth: {node['depth']}")
        lines.append(f"{indent}Is Leaf: {node['is_leaf']}")
        lines.append(f"{indent}Reward: {node.get('reward', 'N/A')}")
        lines.append(f"{indent}Answer: {node.get('answer', 'N/A')}")
        lines.append(f"{indent}Search Query: {node.get('search_query', 'N/A')}")
        
        if node.get('response'):
            lines.append(f"{indent}Response: {node['response'][:200]}...")
        
        if node.get('sequence'):
            lines.append(f"{indent}Sequence Length: {len(node['sequence'])} chars")
        
        lines.append("")
        
        # Add children
        for child in node.get('children', []):
            lines.append(format_node(child, depth + 1))
        
        return "\n".join(lines)
    
    # Create readable content
    content = []
    content.append("=" * 80)
    content.append(f"TRAJECTORY TREE FOR QUESTION: {trajectory_tree['question_id']}")
    content.append("=" * 80)
    content.append(f"Question: {trajectory_tree['question']}")
    content.append(f"Golden Answers: {trajectory_tree['golden_answers']}")
    content.append(f"Total Nodes: {trajectory_tree['total_nodes']}")
    content.append(f"Leaf Nodes: {len(trajectory_tree['leaf_nodes'])}")
    content.append(f"Max Depth: {trajectory_tree['max_depth']}")
    content.append(f"Max First Width: {trajectory_tree['max_first_width']}")
    content.append(f"Max Width: {trajectory_tree['max_width']}")
    content.append("")
    content.append("TREE STRUCTURE:")
    content.append("-" * 40)
    content.append(format_node(trajectory_tree['root']))
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))


def save_trajectories_json(trajectory_tree: Dict[str, Any], output_path: str):
    """
    Save all trajectories (paths from root to leaf) in JSON format.
    
    Args:
        trajectory_tree: The trajectory tree structure
        output_path: Path to save the trajectories
    """
    trajectories = []
    
    for leaf_node in trajectory_tree["leaf_nodes"]:
        # Build trajectory path
        path = []
        current = leaf_node
        while current is not None:
            path.append({
                "id": current["id"],
                "depth": current["depth"],
                "response": current.get("response", ""),
                "search_query": current.get("search_query", ""),
                "answer": current.get("answer", ""),
                "reward": current.get("reward", 0.0),
                "is_leaf": current["is_leaf"]
            })
            current = current.get("parent")
        path.reverse()  # From root to leaf
        
        trajectory_data = {
            "trajectory_id": f"{trajectory_tree['question_id']}_traj_{len(trajectories)}",
            "question_id": trajectory_tree['question_id'],
            "question": trajectory_tree['question'],
            "golden_answers": trajectory_tree['golden_answers'],
            "path": path,
            "final_answer": leaf_node.get("answer", ""),
            "is_correct": leaf_node.get("answer", "") in trajectory_tree['golden_answers'],
            "total_reward": sum(node.get("reward", 0.0) for node in path),
            "path_length": len(path)
        }
        trajectories.append(trajectory_data)
    
    # Save trajectories
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, indent=2, ensure_ascii=False)


def save_training_metadata(args, output_dir: str):
    """
    Save training configuration and metadata.
    
    Args:
        args: Command line arguments
        output_dir: Output directory path
    """
    metadata = {
        "training_start_time": datetime.now().isoformat(),
        "script_name": "trpo.py",
        "arguments": vars(args),
        "environment": {
            "python_version": "3.x",
            "pytorch_version": torch.__version__,
            "device": str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
        }
    }
    
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)





def assign_rewards_to_trajectory_tree(trajectory_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign rewards to all nodes in the trajectory tree based on correctness.
    
    Args:
        trajectory_tree: The trajectory tree structure
        
    Returns:
        Updated trajectory tree with rewards assigned to each node
    """
    golden_answers = trajectory_tree["golden_answers"]
    
    # First pass: identify reward nodes (nodes on paths to correct answers)
    reward_nodes = set()
    
    for leaf_node in trajectory_tree["leaf_nodes"]:
        if leaf_node["answer"] in golden_answers:
            # This leaf leads to a correct answer, mark all nodes on this path as reward nodes
            current = leaf_node
            while current is not None:
                reward_nodes.add(current["id"])
                current = current.get("parent")
    
    # Second pass: assign rewards to all nodes
    def assign_rewards_recursive(node: Dict[str, Any]):
        # Assign reward based on whether this node is on a path to correct answer
        if node["id"] in reward_nodes:
            node["reward"] = 1.0  # Positive reward for nodes leading to correct answers
        else:
            node["reward"] = -1.0  # Negative reward for nodes not leading to correct answers
        
        # Recursively assign rewards to children
        for child in node["children"]:
            assign_rewards_recursive(child)
    
    # Start from root and assign rewards recursively
    assign_rewards_recursive(trajectory_tree["root"])
    
    return trajectory_tree


def compute_masked_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    full: str,
    return_length: bool = False,
) -> torch.Tensor:
    """
    Compute sum of log-probs of continuation tokens conditioned on prompt,
    excluding tokens whose character spans in `continuation` overlap any <information>...</information> spans.
    Returns a single-scalar torch.Tensor (sum of selected token log-probs).
    Gradients flow back to model parameters (do NOT wrap in torch.no_grad()).
    """
    if full is None:
        if return_length:
            return torch.tensor(0.0, device=model.device, requires_grad=True), 0
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    
    # Log memory before starting
    log_cuda_memory("before compute_masked_logprob")
    
    # Tokenize prompt and full text
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    full_tokens = tokenizer.encode(full, add_special_tokens=False)
    
    # Extract continuation tokens (tokens after the prompt)
    if len(full_tokens) <= len(prompt_tokens):
        # No continuation tokens
        if return_length:
            return torch.tensor(0.0, device=model.device, requires_grad=True), 0
        return torch.tensor(0.0, device=model.device, requires_grad=True)
    
    continuation_tokens = full_tokens[len(prompt_tokens):]
    
    # Find information spans in the full text
    info_spans = []
    info_pattern = r'<information>.*?</information>'
    for match in re.finditer(info_pattern, full, re.DOTALL | re.IGNORECASE):
        info_spans.append((match.start(), match.end()))
    
    # Find which continuation tokens to mask
    masked_continuation_indices = set()
    if info_spans:
        # Get character positions of continuation tokens in the full text
        prompt_len = len(tokenizer.decode(prompt_tokens))
        
        for i, token in enumerate(continuation_tokens):
            # Calculate the position of this token in the full text
            token_start_in_full = prompt_len + len(tokenizer.decode(continuation_tokens[:i]))
            token_end_in_full = token_start_in_full + len(tokenizer.decode([token]))
            
            # Check if this token overlaps with any information span
            for span_start, span_end in info_spans:
                if not (token_end_in_full <= span_start or token_start_in_full >= span_end):
                    masked_continuation_indices.add(i)
                    break
    
    # Prepare input for model
    input_ids = torch.tensor([full_tokens], device=model.device)
    
    # Get model outputs (without no_grad to allow gradients)
    try:
        outputs = model(input_ids)
        logits = outputs.logits[0]  # Remove batch dimension
        log_cuda_memory("after model forward pass")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM during model forward pass: {e}")
        log_cuda_memory("OOM during model forward pass")
        cleanup_cuda_memory()
        raise
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Sum log probabilities for continuation tokens (excluding masked ones)
    total_log_prob = torch.tensor(0.0, device=model.device, requires_grad=True)
    num_unmasked_tokens = 0
    for i, token_id in enumerate(continuation_tokens):
        if i not in masked_continuation_indices:
            # Get log probability of this token
            token_log_prob = log_probs[len(prompt_tokens) + i - 1, token_id]
            total_log_prob = total_log_prob + token_log_prob  # Use non-in-place addition
            num_unmasked_tokens += 1
    
    if return_length:
        return total_log_prob, num_unmasked_tokens
    return total_log_prob


def compute_trpo_loss(
    model: torch.nn.Module,
    tokenizer,
    trajectory_tree: Dict[str, Any],
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute TRPO loss for the trajectory tree.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        trajectory_tree: Trajectory tree with rewards assigned
        device: Device to run computations on
        clip_ratio: PPO clipping ratio
        value_coef: Value function loss coefficient
        entropy_coef: Entropy bonus coefficient
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    log_cuda_memory("before compute_trpo_loss")
    total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    total_kl_div = 0.0
    total_entropy = torch.tensor(0.0, device=model.device, requires_grad=True)
    num_trajectories = 0
    
    # Process each trajectory (path from root to leaf)
    logger.debug(f"Processing {len(trajectory_tree['leaf_nodes'])} leaf nodes for TRPO loss computation")
    for i, leaf_node in enumerate(trajectory_tree["leaf_nodes"]):
        logger.debug(f"Processing leaf node {i+1}/{len(trajectory_tree['leaf_nodes'])}: {leaf_node.get('id', 'unknown')}")
        
        # Build trajectory path
        path = []
        current = leaf_node
        while current is not None:
            path.append(current)
            current = current.get("parent")
        path.reverse()  # From root to leaf
        
        logger.debug(f"Trajectory path length: {len(path)}")
        if len(path) < 2:  # Skip if trajectory is too short
            logger.debug(f"Skipping trajectory with path length {len(path)} < 2")
            continue
        
        # Compute trajectory-level metrics
        trajectory_reward = (sum(node.get("reward", 0.0) for node in path) - 1)/(len(path)-1)
        trajectory_text = leaf_node["sequence"]
        
        # Get initial prompt
        initial_prompt = trajectory_tree["root"]["sequence"]
        
        # Compute log probabilities for this trajectory
        try:
            logger.debug(f"Computing log probability for trajectory with text length: {len(trajectory_text)}")
            log_cuda_memory(f"before trajectory {i+1} processing")
            
            log_prob, trajectory_length = compute_masked_logprob(model, tokenizer, initial_prompt, trajectory_text, return_length=True)
            logger.debug(f"Log probability computed successfully: {log_prob}, length: {trajectory_length}")
            
            log_cuda_memory(f"after trajectory {i+1} processing")
            
            # Normalize by trajectory length to prevent bias towards longer sequences
            # This ensures fair comparison between trajectories of different lengths
            logger.debug(f"TRPO: Before normalization: log_prob={log_prob:.6f}, length={trajectory_length}")
            if trajectory_length > 0:
                log_prob = log_prob / trajectory_length
            logger.debug(f"TRPO: After normalization: log_prob={log_prob:.6f}")
            
            # Compute policy loss (simplified TRPO - using PPO-style clipping)
            # In a full TRPO implementation, you would use conjugate gradient and line search
            # Here we use a simplified version with clipping
            
            # For now, use a simple policy gradient loss weighted by rewards
            # Convert trajectory_reward to tensor to ensure proper gradient computation
            reward_tensor = torch.tensor(trajectory_reward, device=model.device, dtype=log_prob.dtype)
            policy_loss = -log_prob * reward_tensor
            
            total_loss = total_loss + policy_loss  # Use non-in-place addition
            total_entropy = total_entropy + (-log_prob)  # Use non-in-place addition
            num_trajectories += 1
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during trajectory {i+1} processing: {e}")
            log_cuda_memory("OOM during trajectory processing")
            cleanup_cuda_memory()
            continue
        except Exception as e:
            logger.warning(f"Error computing log probability for trajectory: {e}")
            continue
        
        # Clean up memory after each trajectory to prevent accumulation

        cleanup_cuda_memory()
    
    if num_trajectories == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True), {}
    
    # Average the losses
    avg_loss = total_loss / num_trajectories
    avg_entropy = total_entropy / num_trajectories
    
    # Add entropy bonus
    entropy_coef_tensor = torch.tensor(entropy_coef, device=model.device, dtype=avg_loss.dtype)
    final_loss = avg_loss - entropy_coef_tensor * avg_entropy
    
    metrics = {
        "policy_loss": avg_loss.item(),
        "entropy": avg_entropy.item(),
        "num_trajectories": num_trajectories,
        "total_loss": final_loss.item()
    }
    
    return final_loss, metrics


def trpo(trajectory_tree: Dict[str, Any], model: torch.nn.Module, tokenizer, optimizer):
    """
    Train using TRPO on the trajectory tree.
    
    Args:
        trajectory_tree: The trajectory tree structure
        model: The language model to train
        tokenizer: The tokenizer
        optimizer: The optimizer
        device: Device to run computations on
        
    Returns:
        Tuple of (updated_model, metrics)
    """
    # Assign rewards to all nodes in the trajectory tree
    trajectory_tree = assign_rewards_to_trajectory_tree(trajectory_tree)
    
    # Compute TRPO loss
    loss, metrics = compute_trpo_loss(model, tokenizer, trajectory_tree)
    
    # Backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    
    # In a full TRPO implementation, you would:
    # 1. Compute natural gradient using conjugate gradient
    # 2. Perform line search to find optimal step size
    # 3. Apply the update with trust region constraints
    
    # For this simplified version, we use standard gradient descent
    # with gradient clipping as a proxy for trust region constraints
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Log metrics
    logger.info(f"TRPO Training - Loss: {metrics['total_loss']:.4f}, "
                f"Policy Loss: {metrics['policy_loss']:.4f}, "
                f"Entropy: {metrics['entropy']:.4f}, "
                f"Trajectories: {metrics['num_trajectories']}")
    
    return model, metrics



# -----------------------------
# CLI
# -----------------------------
def main(args):
    # Load dataset
    items = load_dataset(args.data_path, args.max_data_samples)
    #initialize questions
    questions = initialize_questions(items)
    #load system
    system = load_system(args.reasoner_model_name, args.summarizer_model_name, args.reasoner_lora_path, args.retriever_type, args.retriever_index_path, args.e5_model_path)

    # Get the reasoner model and tokenizer
    model = system.reasoner.model
    tokenizer = system.reasoner.tokenizer
    
    # Set model to training mode for gradient computation
    logger.info(f"Model training mode before setting: {model.training}")
    model.train()
    logger.info(f"Model training mode after setting: {model.training}")
    
    # Ensure all model parameters require gradients
    grad_params = 0
    for param in model.parameters():
        if param.requires_grad:
            grad_params += 1
    logger.info(f"Model parameters requiring gradients: {grad_params}/{sum(1 for _ in model.parameters())}")
    
    for param in model.parameters():
        param.requires_grad = True
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    trajectories_dir = os.path.join(args.output_dir, "trajectories")
    trees_dir = os.path.join(args.output_dir, "trees")
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    models_dir = os.path.join(args.output_dir, "models")
    
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(trees_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save training metadata and create README
    save_training_metadata(args, args.output_dir)
    
    # Training Loop
    logger.info(f"Starting TRPO training on {len(questions)} questions")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Trajectories will be saved to: {trajectories_dir}")
    logger.info(f"Tree structures will be saved to: {trees_dir}")
    logger.info(f"Checkpoints will be saved to: {checkpoints_dir}")
    logger.info(f"Models will be saved to: {models_dir}")
    
    training_metrics = []
    training_start_time = time.time()
    
    for question_idx, question in enumerate(tqdm(questions, desc="Training on questions")):
        logger.info(f"Processing question {question_idx + 1}/{len(questions)}: {question['id']}")
        
        try:
            question_start_time = time.time()
            
            # Test reasoner before generating trajectory tree
            logger.info("Testing reasoner with simple question...")
            test_sequence = f"Question: {question['question']}\n\nPlease think step by step and provide an answer."
            try:
                test_response = system.reasoner.generate_response(test_sequence)
                logger.info(f"Reasoner test successful. Response: {test_response[:100]}...")
            except Exception as e:
                logger.error(f"Reasoner test failed: {e}")
                logger.error(f"Exception type: {type(e)}")
                # Continue anyway to see what happens
            
            # Generate trajectory tree for this question
            logger.info("Generating trajectory tree...")
            trajectory_tree = generate_trajectory_tree(
                question, 
                system, 
                args.max_depth, 
                args.max_first_width, 
                args.max_width
            )
            
            logger.info(f"Generated trajectory tree with {trajectory_tree['total_nodes']} nodes and {len(trajectory_tree['leaf_nodes'])} leaf nodes")
            
            # Initialize file paths
            tree_readable_path = None
            trajectories_json_path = None
            
            # Check if we have any leaf nodes to work with
            if len(trajectory_tree['leaf_nodes']) == 0:
                logger.warning(f"No leaf nodes generated for question {question['id']}. Skipping training for this question.")
                # Create dummy metrics for this question
                metrics = {
                    "policy_loss": 0.0,
                    "entropy": 0.0,
                    "num_trajectories": 0,
                    "total_loss": 0.0
                }
                # Keep the same model (no update)
                updated_model = model
            else:
                logger.info(f"Question {question['id']} has {len(trajectory_tree['leaf_nodes'])} leaf nodes. Proceeding with TRPO training.")
                # Save trajectory tree in readable format
                tree_readable_path = os.path.join(trees_dir, f"question_{question_idx + 1:03d}_{question['id']}_tree.txt")
                save_trajectory_tree_readable(trajectory_tree, tree_readable_path)
                logger.info(f"Saved tree structure to: {tree_readable_path}")
                
                # Save trajectories in JSON format
                trajectories_json_path = os.path.join(trajectories_dir, f"question_{question_idx + 1:03d}_{question['id']}_trajectories.json")
                save_trajectories_json(trajectory_tree, trajectories_json_path)
                logger.info(f"Saved trajectories to: {trajectories_json_path}")
                
            # Train the model using TRPO on this trajectory tree
            logger.info("Training with TRPO...")
            logger.info(f"Model training mode before TRPO: {model.training}")
            log_cuda_memory("before TRPO training")
            
            try:
                updated_model, metrics = trpo(trajectory_tree, model, tokenizer, optimizer)
                log_cuda_memory("after TRPO training")
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM during TRPO training for question {question['id']}: {e}")
                log_cuda_memory("OOM during TRPO training")
                cleanup_cuda_memory()
                # Skip this question and continue
                continue
            
            # Update the model reference
            model = updated_model
            system.reasoner.model = model
            
            # Store metrics with additional information
            metrics['question_id'] = question['id']
            metrics['question_idx'] = question_idx
            metrics['question_text'] = question['question']
            metrics['golden_answers'] = question['golden_answers']
            metrics['trajectory_tree_stats'] = {
                'total_nodes': trajectory_tree['total_nodes'],
                'leaf_nodes': len(trajectory_tree['leaf_nodes']),
                'max_depth': trajectory_tree['max_depth'],
                'max_first_width': trajectory_tree['max_first_width'],
                'max_width': trajectory_tree['max_width']
            }
            metrics['processing_time'] = time.time() - question_start_time
            metrics['tree_output_path'] = tree_readable_path or "N/A"
            metrics['trajectories_output_path'] = trajectories_json_path or "N/A"
            training_metrics.append(metrics)
            
            # Save checkpoint periodically
            if (question_idx + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_question_{question_idx + 1:03d}")
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                
                # Save model state
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                # Save training metrics
                metrics_path = os.path.join(checkpoint_path, "training_metrics.json")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(training_metrics, f, indent=2, ensure_ascii=False)
                
                # Save checkpoint metadata
                checkpoint_metadata = {
                    "checkpoint_number": question_idx + 1,
                    "question_id": question['id'],
                    "timestamp": datetime.now().isoformat(),
                    "total_questions_processed": question_idx + 1,
                    "total_questions": len(questions),
                    "training_time_so_far": time.time() - training_start_time
                }
                metadata_path = os.path.join(checkpoint_path, "checkpoint_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Checkpoint saved successfully")
            
            # Clean up memory after each question
            cleanup_cuda_memory()
        
        except Exception as e:
            logger.error(f"Error processing question {question['id']}: {e}")
            continue
    
    # Save final model
    final_model_path = os.path.join(models_dir, "final_model")
    logger.info(f"Saving final model to {final_model_path}")
    
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save final training metrics
    final_metrics_path = os.path.join(args.output_dir, "final_training_metrics.json")
    with open(final_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=2, ensure_ascii=False)
    
    # Save final training summary
    total_training_time = time.time() - training_start_time
    training_summary = {
        "training_completed_at": datetime.now().isoformat(),
        "total_training_time_seconds": total_training_time,
        "total_training_time_hours": total_training_time / 3600,
        "total_questions_processed": len(training_metrics),
        "total_questions_attempted": len(questions),
        "success_rate": len(training_metrics) / len(questions) if questions else 0,
        "final_model_path": final_model_path,
        "output_directory": args.output_dir,
        "arguments": vars(args)
    }
    
    if training_metrics:
        training_summary.update({
            "average_policy_loss": sum(m['policy_loss'] for m in training_metrics) / len(training_metrics),
            "average_entropy": sum(m['entropy'] for m in training_metrics) / len(training_metrics),
            "total_trajectories": sum(m['num_trajectories'] for m in training_metrics),
            "average_processing_time_per_question": sum(m['processing_time'] for m in training_metrics) / len(training_metrics),
            "average_trajectory_tree_nodes": sum(m['trajectory_tree_stats']['total_nodes'] for m in training_metrics) / len(training_metrics),
            "average_leaf_nodes": sum(m['trajectory_tree_stats']['leaf_nodes'] for m in training_metrics) / len(training_metrics)
        })
    
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Training metrics saved to: {final_metrics_path}")
    logger.info(f"Training summary saved to: {summary_path}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    
    # Print summary statistics
    if training_metrics:
        logger.info(f"Training Summary:")
        logger.info(f"  Questions processed: {len(training_metrics)}/{len(questions)}")
        logger.info(f"  Success rate: {len(training_metrics)/len(questions)*100:.1f}%")
        logger.info(f"  Average policy loss: {training_summary['average_policy_loss']:.4f}")
        logger.info(f"  Average entropy: {training_summary['average_entropy']:.4f}")
        logger.info(f"  Total trajectories: {training_summary['total_trajectories']}")
        logger.info(f"  Average processing time per question: {training_summary['average_processing_time_per_question']:.2f} seconds")
        logger.info(f"  Average trajectory tree nodes: {training_summary['average_trajectory_tree_nodes']:.1f}")
        logger.info(f"  Average leaf nodes: {training_summary['average_leaf_nodes']:.1f}")
    
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info(f"  - Models: {models_dir}")
    logger.info(f"  - Trajectories: {trajectories_dir}")
    logger.info(f"  - Tree structures: {trees_dir}")
    logger.info(f"  - Checkpoints: {checkpoints_dir}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRPO training script for trajectory-based policy optimization")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--max_data_samples", type=int, required=True, help="Maximum number of data samples to use")
    parser.add_argument("--reasoner_model_name", type=str, required=True, help="Base model name for reasoner")
    parser.add_argument("--summarizer_model_name", type=str, required=True, help="Model name for summarizer")
    parser.add_argument("--reasoner_lora_path", type=str, required=True, help="Path to LoRA adapter for reasoner")
    parser.add_argument("--retriever_type", type=str, required=True, help="Type of retriever to use")
    parser.add_argument("--retriever_index_path", type=str, required=True, help="Path to retriever index")
    parser.add_argument("--e5_model_path", type=str, required=True, help="Path to E5 model")
    parser.add_argument("--max_depth", type=int, required=True, help="Maximum depth of trajectory tree")
    parser.add_argument("--max_first_width", type=int, required=True, help="Maximum width at first level")
    parser.add_argument("--max_width", type=int, required=True, help="Maximum width at subsequent levels")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and final model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N steps")
    args = parser.parse_args()

    main(args)
