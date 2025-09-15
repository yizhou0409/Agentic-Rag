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
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

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
        ref_traj = it.get("reference_trajectory") or it.get("ref") or it.get("reference")
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
        'praent': None,
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
    
    # Queue for BFS traversal: (node, parent_depth)
    queue = [(root_node, 0)]
    leaf_nodes = []
    
    while queue:
        current_node, current_depth = queue.pop(0)
        
        if current_depth == max_depth - 1:
            current_node["sequence"] += MAX_TURN_WARNING
        # Generate multiple trajectories from current node

        # Determine how many branches to generate
        if current_depth == 0:
            num_branches = max_first_width
        else:
            num_branches = max_width
            
        # Create multiple copies of the current question for parallel inference
        queries = []
        generated_nodes = 0

        if current_node["search_query"]:
            # Perform retrieval and add to response
            retrieved_docs = system.retriever.retrieve(current_node["search_query"])
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
            info_text = f"\n<information> {summary} </information>\n"
            current_node["sequence"] += info_text

        while generated_nodes <= num_branches:
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
            updated_question, search_query = system.inference_one_turn([question_copy])[0][0], system.inference_one_turn([question_copy])[1][0]
            if search_query in queries:
                continue
            else:
                generated_nodes += 1
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
                if child_node["answer"]:
                    child_node["is_leaf"] = True
                    leaf_nodes.append(child_node)
                elif not child_node["search_query"]:
                    leaf_nodes.append(child_node)
                else:
                    queue.append((child_node, current_depth + 1))
                    queries.append(search_query)
                    current_node["children"].append(child_node)

    
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


def trpo(trajectory_tree: Dict[str, Any]):
    
    # Get golden answers for comparison
    golden_answers = trajectory_tree["golden_answers"]

    correct_trajectories = []
    for leaf_node in trajectory_tree["leaf_nodes"]:
        if leaf_node["answer"] in golden_answers:
            current = leaf_node
            path = []
            while current is not None:
                path.append(current)
                current = current["parent"]
            correct_trajectories.append((leaf_node["sequence"], path))
    
    #TODO: ASSIGNING REWARDS AND TRAINING DRPO

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
    for question in questions:
        #generate trajectory tree
        trajectory_tree = generate_trajectory_tree(question, system, args.max_depth, args.max_first_width, args.max_width)
        #train trpo
        trpo(trajectory_tree)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_data_samples", type=int, required=True)
    parser.add_argument("--reasoner_model_name", type=str, required=True)
    parser.add_argument("--summarizer_model_name", type=str, required=True)
    parser.add_argument("--reasoner_lora_path", type=str, required=True)
    parser.add_argument("--retriever_type", type=str, required=True)
    parser.add_argument("--retriever_index_path", type=str, required=True)
    parser.add_argument("--e5_model_path", type=str, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    parser.add_argument("--max_first_width", type=int, required=True)
    parser.add_argument("--max_width", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)
