#!/usr/bin/env python3
"""
trpo.py

Testing script that:
- uses Reasoner from main.py to sample trajectories (so generation matches inference)
- generates trajectory trees for each question
- evaluates the best result of each trajectory as the metric for each question

Usage example:
python trpo.py \
  --data_path test_data.json \
  --reasoner_model_name /scratch/yl9038/models/Qwen3-0.6B \
  --reasoner_lora_path models/llamafactory/qwen3-lora-0.6B \
  --summarizer_model_name summarizer_model \
  --retriever_type faiss \
  --retriever_index_path retriever_index \
  --e5_model_path e5_model \
  --max_depth 3 \
  --max_first_width 2 \
  --max_width 2 \
  --output_dir test_output

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

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Reasoner and InferenceConfig from your main.py
# main.py must be next to this script and must not execute its main() on import (your main.py uses if __name__ == "__main__":)
from main import  InferenceSystem, InferenceConfig
from utils import is_exact_match, load_dataset, parse_reasoning_generation, calculate_metrics, extract_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# -----------------------------
# Answer extraction / normalization
# -----------------------------
# Note: Using utils.parse_reasoning_generation() for comprehensive response parsing
# and utils.extract_answer() as fallback for better answer extraction

MAX_TURN_WARNING = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."

def load_dataset_questions(dataset_name: str, split: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load questions from a dataset file.
    
    Args:
        dataset_name: Name of the dataset (hotpotqa, 2wikimultihop)
        split: Dataset split (train, dev, test)
        max_samples: Maximum number of samples to load
        
    Returns:
        List of question dictionaries
    """
    dataset_path = f"data/{dataset_name}/{split}.jsonl"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    questions = []
    logger.info(f"Loading dataset from: {dataset_path}")
    
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
                    "golden_answers": [data["answer"]]  # Convert to list format
                })
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    logger.info(f"Loaded {len(questions)} questions from {dataset_name} {split} split")
    return questions

def load_system(reasoner_model_name: str, summarizer_model_name: str, reasoner_lora_path: Optional[str], retriever_type: str, retriever_index_path: str, e5_model_path: str, high_randomness: bool = False) -> InferenceSystem:
    config = InferenceConfig(
        reasoner_model_name=reasoner_model_name,
        summarizer_model_name=summarizer_model_name,
        reasoner_lora_path=reasoner_lora_path,
        retriever_type=retriever_type,
        retriever_index_path=retriever_index_path,
        e5_model_path=e5_model_path,
        high_randomness_mode=high_randomness
    )
    return InferenceSystem(config)

def _count_nodes(node: Dict[str, Any]) -> int:
    """Count total nodes in the trajectory tree."""
    count = 1  # Count current node
    for child in node["children"]:
        count += _count_nodes(child)
    return count

def generate_trajectory_tree(question: Dict[str, Any], system: InferenceSystem, max_depth: int, max_first_width: int, max_width: int, max_branching_depth: int = None) -> Dict[str, Any]:
    """
    Generate a tree of trajectories for a given question using the InferenceSystem.
    
    Args:
        question: Question data with 'id', 'question', 'golden_answers'
        system: InferenceSystem instance for generating responses
        max_depth: Maximum depth of the trajectory tree
        max_first_width: Maximum number of branches at the first level
        max_width: Maximum number of branches at subsequent levels
        max_branching_depth: Maximum depth at which to stop generating branches (default: max_depth-1)
        
    Returns:
        Dictionary containing the trajectory tree structure
    """
    # Set default max_branching_depth if not provided
    if max_branching_depth is None:
        max_branching_depth = max_depth
    
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
        # Stop generating branches if we've reached max_branching_depth
        if current_depth >= max_branching_depth:
            num_branches = 1  # No more branches at this depth
        elif current_depth == 0:
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
        "max_branching_depth": max_branching_depth,
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
    content.append(f"Max Branching Depth: {trajectory_tree['max_branching_depth']}")
    content.append("")
    content.append("TREE STRUCTURE:")
    content.append("-" * 40)
    content.append(format_node(trajectory_tree['root']))
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))


def save_trajectories_json(trajectory_tree: Dict[str, Any], output_path: str):
    """
    Save only the best trajectory (highest F1 score) in JSON format.
    All trajectories are still preserved in the tree structure.
    
    Args:
        trajectory_tree: The trajectory tree structure
        output_path: Path to save the best trajectory
    """
    trajectories = []
    best_f1_score = -1.0
    best_trajectory = None
    
    for leaf_node in trajectory_tree["leaf_nodes"]:
        # Parse response using utils.parse_reasoning_generation() for better extraction
        response_text = leaf_node.get("response", "")
        thought, parsed_search_query, parsed_answer = parse_reasoning_generation(response_text)
        
        # Build trajectory path
        path = []
        current = leaf_node
        while current is not None:
            # Parse each node's response for better extraction
            node_response = current.get("response", "")
            node_thought, node_parsed_search_query, node_parsed_answer = parse_reasoning_generation(node_response)
            
            
            path.append({
                "id": current["id"],
                "depth": current["depth"],
                "response": node_response,
                "thought": node_thought,
                "search_query": node_parsed_search_query,
                "answer": node_parsed_answer,
                "reward": current.get("reward", 0.0),
                "is_leaf": current["is_leaf"]
            })
            current = current.get("parent")
        path.reverse()  # From root to leaf
        
        # Calculate F1 score for this trajectory
        f1_score = calculate_metrics(parsed_answer, trajectory_tree['golden_answers'])["f1"]
        
        # Use utils.is_exact_match for case-insensitive comparison
        is_correct = any(is_exact_match(parsed_answer, ga) for ga in trajectory_tree['golden_answers'])
        
        trajectory_data = {
            "trajectory_id": f"{trajectory_tree['question_id']}_traj_{len(trajectories)}",
            "question_id": trajectory_tree['question_id'],
            "question": trajectory_tree['question'],
            "golden_answers": trajectory_tree['golden_answers'],
            "path": path,
            "final_answer": parsed_answer,
            "is_correct": is_correct,
            "f1": f1_score,
            "total_reward": sum(node.get("reward", 0.0) for node in path),
            "path_length": len(path)
        }
        trajectories.append(trajectory_data)
        
        # Track the best trajectory
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_trajectory = trajectory_data
    
    # Save only the best trajectory (or empty list if no trajectories)
    best_trajectories = [best_trajectory] if best_trajectory else []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(best_trajectories, f, indent=2, ensure_ascii=False)







def generate_n_trajectories(question: Dict[str, Any], system: InferenceSystem, n_trajectories: int, max_turns: int = 5) -> List[Dict[str, Any]]:
    """
    Generate n independent trajectories for a given question using the InferenceSystem.
    Each trajectory is generated independently with high randomness settings for diversity,
    and the best one is selected based on F1 score.
    
    Args:
        question: Question data with 'id', 'question', 'golden_answers'
        system: InferenceSystem instance for generating responses (with high_randomness_mode=True)
        n_trajectories: Number of trajectories to generate
        max_turns: Maximum number of turns per trajectory
        
    Returns:
        List of trajectory results, each containing the full conversation path
    """
    trajectories = []
    
    logger.info(f"Generating {n_trajectories} trajectories for question {question['id']}")
    
    for traj_idx in range(n_trajectories):
        logger.info(f"Generating trajectory {traj_idx + 1}/{n_trajectories}")
        
        try:
            # Initialize sequence with the question under prompt template
            prompted_question = system.reasoner.prompt_template.format(question=question["question"])
            
            # Prepare messages for chat template
            messages = [{"role": "user", "content": prompted_question}]
            initial_sequence = system.reasoner.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            # Initialize trajectory data
            trajectory_data = {
                "id": f"{question['id']}_traj_{traj_idx}",
                "question": question["question"],
                "golden_answers": question["golden_answers"],
                "sequence": initial_sequence,
                "response": "",
                "turns": [],
                "final_turn": 0,
                "answer": None,
                "error": None,
                "trajectory_idx": traj_idx
            }
            
            # Generate trajectory using the inference system
            max_turn_warning = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."
            
            for turn_num in range(max_turns + 1):
                # Generate response for this turn
                response = system.reasoner.generate_response(trajectory_data["sequence"])
                
                # Extract search query or answer
                search_query = system._extract_search_query(response)
                answer = system._extract_answer(response)
                
                # Record turn info
                turn_info = {
                    "turn": turn_num + 1,
                    "response": response,
                    "search_query": search_query,
                    "answer": answer
                }
                trajectory_data["turns"].append(turn_info)
                
                # Update sequence with the current response
                trajectory_data["sequence"] += response
                trajectory_data["response"] += response
                
                if answer:
                    # Question answered
                    trajectory_data["answer"] = answer
                    trajectory_data["final_turn"] = turn_num + 1
                    logger.info(f"Trajectory {traj_idx + 1} completed in turn {turn_num + 1}")
                    break
                elif search_query:
                    # Need to search, perform retrieval
                    try:
                        retrieved_docs = system.retriever.search(search_query, num=system.config.top_k_docs)
                        doc_texts = []
                        for doc in retrieved_docs:
                            if isinstance(doc, dict):
                                doc_text = doc.get('text', doc.get('contents', str(doc)))
                            else:
                                doc_text = str(doc)
                            doc_texts.append(doc_text)
                        
                        # Summarize the retrieved documents
                        summary = system.summarizer.summarize_documents(search_query, doc_texts)
                        
                        # Add summarized information to the sequence
                        info_text = f"\n<information> {summary} </information>"
                        trajectory_data["sequence"] += info_text
                        trajectory_data["response"] += info_text
                        
                        # Update turn info with retrieval results
                        turn_info["retrieved_docs"] = [str(doc) for doc in retrieved_docs]
                        turn_info["summary"] = summary
                        
                    except Exception as e:
                        logger.warning(f"Error in retrieval for trajectory {traj_idx + 1}: {e}")
                        # Continue without retrieval
                else:
                    # No search query or answer found
                    if turn_num == max_turns - 1:
                        trajectory_data["sequence"] += max_turn_warning
                    elif turn_num == max_turns:
                        trajectory_data["error"] = "Max turns reached"
                        trajectory_data["final_turn"] = max_turns
                        break
            
            # Calculate metrics for this trajectory
            if trajectory_data["answer"]:
                from utils import calculate_metrics
                metrics = calculate_metrics(trajectory_data["answer"], trajectory_data["golden_answers"])
                trajectory_data["metrics"] = metrics
            else:
                trajectory_data["metrics"] = {"em": 0.0, "f1": 0.0, "cover_match": 0.0}
            
            trajectories.append(trajectory_data)
            
        except Exception as e:
            logger.error(f"Error generating trajectory {traj_idx + 1} for question {question['id']}: {e}")
            # Create error trajectory
            error_trajectory = {
                "id": f"{question['id']}_traj_{traj_idx}",
                "question": question["question"],
                "golden_answers": question["golden_answers"],
                "sequence": "",
                "response": "",
                "turns": [],
                "final_turn": 0,
                "answer": None,
                "error": str(e),
                "trajectory_idx": traj_idx,
                "metrics": {"em": 0.0, "f1": 0.0, "cover_match": 0.0}
            }
            trajectories.append(error_trajectory)
    
    return trajectories


def select_best_trajectory(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the best trajectory based on F1 score.
    If multiple trajectories have the same F1 score, choose the shorter one.
    
    Args:
        trajectories: List of trajectory results
        
    Returns:
        Dictionary containing the best trajectory and selection metrics
    """
    if not trajectories:
        return {
            "best_trajectory": None,
            "best_f1": 0.0,
            "total_trajectories": 0,
            "average_f1": 0.0,
            "selection_metrics": {}
        }
    
    # Find trajectory with highest F1 score, and if tied, choose the shorter one
    best_trajectory = None
    best_f1 = -1.0
    best_trajectory_length = float('inf')
    f1_scores = []
    
    for trajectory in trajectories:
        f1_score = trajectory.get("metrics", {}).get("f1", 0.0)
        f1_scores.append(f1_score)
        
        # Calculate trajectory length (number of turns)
        trajectory_length = len(trajectory.get("turns", []))
        
        # Select best trajectory: higher F1 score, or if tied, shorter trajectory
        if (f1_score > best_f1 or 
            (f1_score == best_f1 and trajectory_length < best_trajectory_length)):
            best_f1 = f1_score
            best_trajectory_length = trajectory_length
            best_trajectory = trajectory
    
    # Calculate statistics
    total_trajectories = len(trajectories)
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    selection_metrics = {
        "best_f1": best_f1,
        "average_f1": average_f1,
        "total_trajectories": total_trajectories,
        "f1_scores": f1_scores,
        "best_trajectory_idx": best_trajectory.get("trajectory_idx", -1) if best_trajectory else -1,
        "best_trajectory_length": best_trajectory_length
    }
    
    return {
        "best_trajectory": best_trajectory,
        "best_f1": best_f1,
        "total_trajectories": total_trajectories,
        "average_f1": average_f1,
        "selection_metrics": selection_metrics
    }


def evaluate_trajectory_metrics(trajectory_tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate metrics for each trajectory in the trajectory tree.
    The metric for each question is the best F1 result of the trajectory.
    
    Args:
        trajectory_tree: The trajectory tree structure
        
    Returns:
        Dictionary containing evaluation metrics
    """
    golden_answers = trajectory_tree["golden_answers"]
    leaf_nodes = trajectory_tree["leaf_nodes"]
    
    if not leaf_nodes:
        return {
            "question_id": trajectory_tree["question_id"],
            "total_trajectories": 0,
            "best_f1": 0.0,
            "best_result": 0.0,
            "average_f1": 0.0,
            "trajectory_results": []
        }
    
    trajectory_results = []
    f1_scores = []
    
    # Evaluate each trajectory (path from root to leaf)
    for i, leaf_node in enumerate(leaf_nodes):
        # Parse response using utils.parse_reasoning_generation() for better extraction
        response_text = leaf_node.get("response", "")
        thought, parsed_search_query, parsed_answer = parse_reasoning_generation(response_text)
        
        # Use parsed answer if available, otherwise fallback to extract_answer()
        final_answer = parsed_answer if parsed_answer else extract_answer(response_text)
        
        # Calculate F1 score for this trajectory using utils.calculate_metrics()
        if golden_answers:
            metrics = calculate_metrics(final_answer, golden_answers)
            f1_score = metrics["f1"]
        else:
            f1_score = 0.0
        
        f1_scores.append(f1_score)
        
        # Build trajectory path for analysis
        path = []
        current = leaf_node
        while current is not None:
            # Parse each node's response for better extraction
            node_response = current.get("response", "")
            node_thought, node_parsed_search_query, node_parsed_answer = parse_reasoning_generation(node_response)
            
            # Use parsed answer if available, otherwise fallback to extract_answer()
            node_answer = node_parsed_answer if node_parsed_answer else extract_answer(node_response)
            
            path.append({
                "id": current["id"],
                "depth": current["depth"],
                "answer": node_answer,
                "thought": node_thought,
                "search_query": current.get("search_query", "") or node_parsed_search_query
            })
            current = current.get("parent")
        path.reverse()  # From root to leaf
        
        trajectory_result = {
            "trajectory_id": i,
            "final_answer": final_answer,
            "f1_score": f1_score,
            "path_length": len(path),
            "path": path
        }
        trajectory_results.append(trajectory_result)
    
    # The best result is the highest F1 score among all trajectories
    best_f1 = max(f1_scores) if f1_scores else 0.0
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return {
        "question_id": trajectory_tree["question_id"],
        "question": trajectory_tree["question"],
        "golden_answers": golden_answers,
        "total_trajectories": len(leaf_nodes),
        "best_f1": best_f1,
        "best_result": best_f1,  # The best result is the best F1 score
        "average_f1": average_f1,
        "trajectory_results": trajectory_results
    }





# -----------------------------
# N-Trajectory Main Function
# -----------------------------
def main_n_trajectories(args):
    """
    Main function for n-trajectory approach: generates n trajectories for each question
    and selects the best one based on F1 score.
    """
    # Load dataset questions
    questions = load_dataset_questions(args.dataset_name, args.split, args.max_data_samples)
    # Load system with high randomness mode for n_trajectories
    system = load_system(args.reasoner_model_name, args.summarizer_model_name, args.reasoner_lora_path, args.retriever_type, args.retriever_index_path, args.e5_model_path, high_randomness=True)

    # Set model to evaluation mode (no training)
    model = system.reasoner.model
    model.eval()
    logger.info(f"Model set to evaluation mode: {not model.training}")
    logger.info("High randomness mode enabled for diverse trajectory generation")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    trajectories_dir = os.path.join(args.output_dir, "trajectories")
    results_dir = os.path.join(args.output_dir, "results")
    
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Testing Loop
    logger.info(f"Starting n-trajectory testing on {len(questions)} questions")
    logger.info(f"Generating {args.n_trajectories} trajectories per question")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Trajectories will be saved to: {trajectories_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    test_results = []
    testing_start_time = time.time()
    
    for question_idx, question in enumerate(tqdm(questions, desc="Testing questions")):
        logger.info(f"Processing question {question_idx + 1}/{len(questions)}: {question['id']}")
        
        try:
            question_start_time = time.time()
            
            # Generate n trajectories for this question
            logger.info(f"Generating {args.n_trajectories} trajectories...")
            trajectories = generate_n_trajectories(
                question, 
                system, 
                args.n_trajectories,
                args.max_turns
            )
            
            logger.info(f"Generated {len(trajectories)} trajectories")
            
            # Select the best trajectory based on F1 score
            logger.info("Selecting best trajectory...")
            selection_result = select_best_trajectory(trajectories)
            best_trajectory = selection_result["best_trajectory"]
            
            # Save all trajectories for this question
            trajectories_json_path = os.path.join(trajectories_dir, f"question_{question_idx + 1:03d}_{question['id']}_trajectories.json")
            with open(trajectories_json_path, 'w', encoding='utf-8') as f:
                json.dump(trajectories, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved all trajectories to: {trajectories_json_path}")
            
            # Create metrics for this question - save only the best trajectory
            if best_trajectory:
                # Create a clean result similar to main.py format, containing only the best trajectory
                best_result = {
                    "id": question["id"],
                    "question": question["question"],
                    "golden_answers": question["golden_answers"],
                    "sequence": best_trajectory.get("sequence", ""),
                    "response": best_trajectory.get("response", ""),
                    "turns": best_trajectory.get("turns", []),
                    "final_turn": best_trajectory.get("final_turn", 0),
                    "answer": best_trajectory.get("answer"),
                    "error": best_trajectory.get("error"),
                    "metrics": best_trajectory.get("metrics", {"em": 0.0, "f1": 0.0, "cover_match": 0.0}),
                    "trajectory_idx": best_trajectory.get("trajectory_idx", -1),
                    "total_trajectories": len(trajectories),
                    "selection_metrics": selection_result["selection_metrics"],
                    "processing_time": time.time() - question_start_time,
                    "trajectories_output_path": trajectories_json_path
                }
            else:
                # Create error result if no best trajectory found
                best_result = {
                    "id": question["id"],
                    "question": question["question"],
                    "golden_answers": question["golden_answers"],
                    "sequence": "",
                    "response": "",
                    "turns": [],
                    "final_turn": 0,
                    "answer": None,
                    "error": "No valid trajectories generated",
                    "metrics": {"em": 0.0, "f1": 0.0, "cover_match": 0.0},
                    "trajectory_idx": -1,
                    "total_trajectories": len(trajectories),
                    "selection_metrics": selection_result["selection_metrics"],
                    "processing_time": time.time() - question_start_time,
                    "trajectories_output_path": trajectories_json_path
                }
            
            test_results.append(best_result)
            
            # Log results for this question
            logger.info(f"Question {question['id']} Results:")
            logger.info(f"  Best F1 result: {best_result['metrics']['f1']:.4f}")
            logger.info(f"  Best trajectory index: {best_result['trajectory_idx']}")
            logger.info(f"  Total trajectories: {best_result['total_trajectories']}")
            logger.info(f"  Processing time: {best_result['processing_time']:.2f} seconds")
            
            # Clean up memory after each question
            cleanup_cuda_memory()
        
        except Exception as e:
            logger.error(f"Error processing question {question['id']}: {e}")
            # Create error result for this question
            error_result = {
                "id": question['id'],
                "question": question['question'],
                "golden_answers": question['golden_answers'],
                "sequence": "",
                "response": "",
                "turns": [],
                "final_turn": 0,
                "answer": None,
                "error": str(e),
                "metrics": {"em": 0.0, "f1": 0.0, "cover_match": 0.0},
                "trajectory_idx": -1,
                "total_trajectories": 0,
                "selection_metrics": {},
                "processing_time": time.time() - question_start_time if 'question_start_time' in locals() else 0.0,
                "trajectories_output_path": ""
            }
            test_results.append(error_result)
            continue
    
    # Save final test results
    final_results_path = os.path.join(results_dir, "test_results.json")
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # Save final testing summary
    total_testing_time = time.time() - testing_start_time
    testing_summary = {
        "testing_completed_at": datetime.now().isoformat(),
        "total_testing_time_seconds": total_testing_time,
        "total_testing_time_hours": total_testing_time / 3600,
        "total_questions_processed": len(test_results),
        "total_questions_attempted": len(questions),
        "success_rate": len(test_results) / len(questions) if questions else 0,
        "n_trajectories_per_question": args.n_trajectories,
        "max_turns_per_trajectory": args.max_turns,
        "output_directory": args.output_dir,
        "arguments": vars(args)
    }
    
    if test_results:
        # Calculate overall metrics
        total_questions_with_results = len([r for r in test_results if r.get('total_trajectories', 0) > 0])
        total_trajectories = sum(r.get('total_trajectories', 0) for r in test_results)
        best_f1_scores = [r.get('metrics', {}).get('f1', 0.0) for r in test_results if r.get('total_trajectories', 0) > 0]
        best_em_scores = [r.get('metrics', {}).get('em', 0.0) for r in test_results if r.get('total_trajectories', 0) > 0]
        best_cover_match_scores = [r.get('metrics', {}).get('cover_match', 0.0) for r in test_results if r.get('total_trajectories', 0) > 0]
        
        testing_summary.update({
            "questions_with_trajectories": total_questions_with_results,
            "total_trajectories": total_trajectories,
            "overall_best_f1": max(best_f1_scores) if best_f1_scores else 0.0,
            "average_best_f1": sum(best_f1_scores) / len(best_f1_scores) if best_f1_scores else 0.0,
            "overall_best_em": max(best_em_scores) if best_em_scores else 0.0,
            "average_best_em": sum(best_em_scores) / len(best_em_scores) if best_em_scores else 0.0,
            "overall_best_cover_match": max(best_cover_match_scores) if best_cover_match_scores else 0.0,
            "average_best_cover_match": sum(best_cover_match_scores) / len(best_cover_match_scores) if best_cover_match_scores else 0.0,
            "average_processing_time_per_question": sum(r.get('processing_time', 0) for r in test_results) / len(test_results),
            "average_trajectories_per_question": sum(r.get('total_trajectories', 0) for r in test_results) / len(test_results)
        })
    
    summary_path = os.path.join(args.output_dir, "testing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(testing_summary, f, indent=2, ensure_ascii=False)
    
    logger.info("N-trajectory testing completed successfully!")
    logger.info(f"Test results saved to: {final_results_path}")
    logger.info(f"Testing summary saved to: {summary_path}")
    logger.info(f"Total testing time: {total_testing_time:.2f} seconds ({total_testing_time/3600:.2f} hours)")
    
    # Print summary statistics
    if test_results:
        logger.info(f"Testing Summary:")
        logger.info(f"  Questions processed: {len(test_results)}/{len(questions)}")
        logger.info(f"  Success rate: {len(test_results)/len(questions)*100:.1f}%")
        logger.info(f"  Questions with trajectories: {testing_summary.get('questions_with_trajectories', 0)}")
        logger.info(f"  Total trajectories: {testing_summary.get('total_trajectories', 0)}")
        logger.info(f"  Overall best F1: {testing_summary.get('overall_best_f1', 0):.4f}")
        logger.info(f"  Average best F1: {testing_summary.get('average_best_f1', 0):.4f}")
        logger.info(f"  Overall best EM: {testing_summary.get('overall_best_em', 0):.4f}")
        logger.info(f"  Average best EM: {testing_summary.get('average_best_em', 0):.4f}")
        logger.info(f"  Overall best Cover Match: {testing_summary.get('overall_best_cover_match', 0):.4f}")
        logger.info(f"  Average best Cover Match: {testing_summary.get('average_best_cover_match', 0):.4f}")
        logger.info(f"  Average processing time per question: {testing_summary.get('average_processing_time_per_question', 0):.2f} seconds")
        logger.info(f"  Average trajectories per question: {testing_summary.get('average_trajectories_per_question', 0):.1f}")
    
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info(f"  - Results: {results_dir}")
    logger.info(f"  - Trajectories: {trajectories_dir}")


# -----------------------------
# CLI
# -----------------------------
def main(args):
    # Load dataset questions
    questions = load_dataset_questions(args.dataset_name, args.split, args.max_data_samples)
    # Load system
    system = load_system(args.reasoner_model_name, args.summarizer_model_name, args.reasoner_lora_path, args.retriever_type, args.retriever_index_path, args.e5_model_path)

    # Set model to evaluation mode (no training)
    model = system.reasoner.model
    model.eval()
    logger.info(f"Model set to evaluation mode: {not model.training}")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    trajectories_dir = os.path.join(args.output_dir, "trajectories")
    trees_dir = os.path.join(args.output_dir, "trees")
    results_dir = os.path.join(args.output_dir, "results")
    
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(trees_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Testing Loop
    logger.info(f"Starting trajectory testing on {len(questions)} questions")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Trajectories will be saved to: {trajectories_dir}")
    logger.info(f"Tree structures will be saved to: {trees_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    test_results = []
    testing_start_time = time.time()
    
    for question_idx, question in enumerate(tqdm(questions, desc="Testing questions")):
        logger.info(f"Processing question {question_idx + 1}/{len(questions)}: {question['id']}")
        
        try:
            question_start_time = time.time()
            
            # Generate trajectory tree for this question
            logger.info("Generating trajectory tree...")
            trajectory_tree = generate_trajectory_tree(
                question, 
                system, 
                args.max_depth, 
                args.max_first_width, 
                args.max_width,
                args.max_branching_depth
            )
            
            logger.info(f"Generated trajectory tree with {trajectory_tree['total_nodes']} nodes and {len(trajectory_tree['leaf_nodes'])} leaf nodes")
            
            # Save trajectory tree in readable format
            tree_readable_path = os.path.join(trees_dir, f"question_{question_idx + 1:03d}_{question['id']}_tree.txt")
            save_trajectory_tree_readable(trajectory_tree, tree_readable_path)
            logger.info(f"Saved tree structure to: {tree_readable_path}")
            
            # Save best trajectory in JSON format (all trajectories preserved in tree structure)
            trajectories_json_path = os.path.join(trajectories_dir, f"question_{question_idx + 1:03d}_{question['id']}_trajectories.json")
            save_trajectories_json(trajectory_tree, trajectories_json_path)
            logger.info(f"Saved best trajectory to: {trajectories_json_path}")
                
            # Evaluate metrics for this trajectory tree
            logger.info("Evaluating trajectory metrics...")
            metrics = evaluate_trajectory_metrics(trajectory_tree)
            
            # Store metrics with additional information
            metrics['question_idx'] = question_idx
            metrics['processing_time'] = time.time() - question_start_time
            metrics['tree_output_path'] = tree_readable_path
            metrics['trajectories_output_path'] = trajectories_json_path
            metrics['trajectory_tree_stats'] = {
                'total_nodes': trajectory_tree['total_nodes'],
                'leaf_nodes': len(trajectory_tree['leaf_nodes']),
                'max_depth': trajectory_tree['max_depth'],
                'max_first_width': trajectory_tree['max_first_width'],
                'max_width': trajectory_tree['max_width'],
                'max_branching_depth': trajectory_tree['max_branching_depth']
            }
            test_results.append(metrics)
            
            # Log results for this question
            logger.info(f"Question {question['id']} Results:")
            logger.info(f"  Best F1 result: {metrics['best_result']:.4f}")
            logger.info(f"  Average F1: {metrics['average_f1']:.4f}")
            logger.info(f"  Best F1 score: {metrics['best_f1']:.4f}")
            logger.info(f"  Total trajectories: {metrics['total_trajectories']}")
            logger.info(f"  Processing time: {metrics['processing_time']:.2f} seconds")
            
            # Clean up memory after each question
            cleanup_cuda_memory()
        
        except Exception as e:
            logger.error(f"Error processing question {question['id']}: {e}")
            # Create error metrics for this question
            error_metrics = {
                "question_id": question['id'],
                "question": question['question'],
                "golden_answers": question['golden_answers'],
                "question_idx": question_idx,
                "total_trajectories": 0,
                "best_f1": 0.0,
                "best_result": 0.0,
                "average_f1": 0.0,
                "trajectory_results": [],
                "error": str(e),
                "processing_time": time.time() - question_start_time if 'question_start_time' in locals() else 0.0
            }
            test_results.append(error_metrics)
            continue
    
    # Save final test results
    final_results_path = os.path.join(results_dir, "test_results.json")
    with open(final_results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # Save final testing summary
    total_testing_time = time.time() - testing_start_time
    testing_summary = {
        "testing_completed_at": datetime.now().isoformat(),
        "total_testing_time_seconds": total_testing_time,
        "total_testing_time_hours": total_testing_time / 3600,
        "total_questions_processed": len(test_results),
        "total_questions_attempted": len(questions),
        "success_rate": len(test_results) / len(questions) if questions else 0,
        "output_directory": args.output_dir,
        "arguments": vars(args)
    }
    
    if test_results:
        # Calculate overall metrics
        total_questions_with_results = len([r for r in test_results if r.get('total_trajectories', 0) > 0])
        total_trajectories = sum(r.get('total_trajectories', 0) for r in test_results)
        best_f1_scores = [r.get('best_f1', 0.0) for r in test_results if r.get('total_trajectories', 0) > 0]
        average_f1_scores = [r.get('average_f1', 0.0) for r in test_results if r.get('total_trajectories', 0) > 0]
        
        testing_summary.update({
            "questions_with_trajectories": total_questions_with_results,
            "total_trajectories": total_trajectories,
            "overall_best_f1": max(best_f1_scores) if best_f1_scores else 0.0,
            "average_best_f1": sum(best_f1_scores) / len(best_f1_scores) if best_f1_scores else 0.0,
            "overall_average_f1": sum(average_f1_scores) / len(average_f1_scores) if average_f1_scores else 0.0,
            "average_processing_time_per_question": sum(r.get('processing_time', 0) for r in test_results) / len(test_results),
            "average_trajectory_tree_nodes": sum(r.get('trajectory_tree_stats', {}).get('total_nodes', 0) for r in test_results) / len(test_results),
            "average_leaf_nodes": sum(r.get('trajectory_tree_stats', {}).get('leaf_nodes', 0) for r in test_results) / len(test_results)
        })
    
    summary_path = os.path.join(args.output_dir, "testing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(testing_summary, f, indent=2, ensure_ascii=False)
    
    logger.info("Testing completed successfully!")
    logger.info(f"Test results saved to: {final_results_path}")
    logger.info(f"Testing summary saved to: {summary_path}")
    logger.info(f"Total testing time: {total_testing_time:.2f} seconds ({total_testing_time/3600:.2f} hours)")
    
    # Print summary statistics
    if test_results:
        logger.info(f"Testing Summary:")
        logger.info(f"  Questions processed: {len(test_results)}/{len(questions)}")
        logger.info(f"  Success rate: {len(test_results)/len(questions)*100:.1f}%")
        logger.info(f"  Questions with trajectories: {testing_summary.get('questions_with_trajectories', 0)}")
        logger.info(f"  Total trajectories: {testing_summary.get('total_trajectories', 0)}")
        logger.info(f"  Overall best F1: {testing_summary.get('overall_best_f1', 0):.4f}")
        logger.info(f"  Average best F1: {testing_summary.get('average_best_f1', 0):.4f}")
        logger.info(f"  Overall average F1: {testing_summary.get('overall_average_f1', 0):.4f}")
        logger.info(f"  Average processing time per question: {testing_summary.get('average_processing_time_per_question', 0):.2f} seconds")
        logger.info(f"  Average trajectory tree nodes: {testing_summary.get('average_trajectory_tree_nodes', 0):.1f}")
        logger.info(f"  Average leaf nodes: {testing_summary.get('average_leaf_nodes', 0):.1f}")
    
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info(f"  - Results: {results_dir}")
    logger.info(f"  - Trajectories: {trajectories_dir}")
    logger.info(f"  - Tree structures: {trees_dir}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRPO testing script for trajectory-based evaluation")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (hotpotqa, 2wikimultihop)")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split (train, dev, test)")
    parser.add_argument("--max_data_samples", type=int, required=True, help="Maximum number of data samples to use")
    parser.add_argument("--reasoner_model_name", type=str, required=True, help="Base model name for reasoner")
    parser.add_argument("--summarizer_model_name", type=str, required=True, help="Model name for summarizer")
    parser.add_argument("--reasoner_lora_path", type=str, required=True, help="Path to LoRA adapter for reasoner")
    parser.add_argument("--retriever_type", type=str, required=True, help="Type of retriever to use")
    parser.add_argument("--retriever_index_path", type=str, required=True, help="Path to retriever index")
    parser.add_argument("--e5_model_path", type=str, required=True, help="Path to E5 model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for test results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Method selection
    parser.add_argument("--method", type=str, choices=["tree", "n_trajectories"], default="tree", 
                       help="Method to use: 'tree' for trajectory tree generation, 'n_trajectories' for n independent trajectories")
    
    # Tree method parameters
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth of trajectory tree (for tree method)")
    parser.add_argument("--max_first_width", type=int, default=2, help="Maximum width at first level (for tree method)")
    parser.add_argument("--max_width", type=int, default=2, help="Maximum width at subsequent levels (for tree method)")
    parser.add_argument("--max_branching_depth", type=int, default=None, help="Maximum depth at which to stop generating branches (for tree method). If not specified, defaults to max_depth-1")
    
    # N-trajectory method parameters
    parser.add_argument("--n_trajectories", type=int, default=5, help="Number of independent trajectories to generate (for n_trajectories method)")
    parser.add_argument("--max_turns", type=int, default=5, help="Maximum number of turns per trajectory (for n_trajectories method)")
    
    args = parser.parse_args()

    if args.method == "n_trajectories":
        main_n_trajectories(args)
    else:
        main(args)
