#!/usr/bin/env python3
"""
train_dpo.py

DPO training script that:
- uses Reasoner from main.py to sample trajectories (so generation matches inference)
- masks tokens inside <information>...</information> in continuations when computing log-probs
- constructs (pos, neg) pairs per your rules and trains via DPO objective

Usage example:
python train_dpo.py \
  --data dpo_training_data.json \
  --base-model /scratch/yl9038/models/Qwen3-0.6B \
  --lora-adapter models/llamafactory/qwen3-lora-0.6B \
  --output-dir dpo_out \
  --device cuda \
  --epochs 1

"""

import argparse
import json
import logging
import os
import re
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import PeftModel
from accelerate import Accelerator

# Import Reasoner and InferenceConfig from your main.py
# main.py must be next to this script and must not execute its main() on import (your main.py uses if __name__ == "__main__":)
from main import InferenceSystem, InferenceConfig
from utils import load_dataset, is_exact_match
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# -----------------------------
# Answer extraction / normalization
# -----------------------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def extract_answer_from_trajectory(text: str) -> str:
    """Extract content inside <answer>...</answer>. If missing, fallback to last non-empty line."""
    if not text:
        return ""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def normalize_text(s: str) -> str:
    """Simple normalization for exact-match (lower + remove punctuation/articles)."""
    if s is None:
        return ""
    s = s.lower()
    # remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # whitespace collapse
    s = " ".join(s.split())
    return s.strip()



# -----------------------------
# Masked log-prob calculation
# -----------------------------
def compute_masked_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute sum of log-probs of continuation tokens conditioned on prompt,
    excluding tokens whose character spans in `continuation` overlap any <information>...</information> spans.
    Returns a single-scalar torch.Tensor (sum of selected token log-probs).
    Gradients flow back to model parameters (do NOT wrap in torch.no_grad()).
    """
    if continuation is None:
        continuation = ""
    if continuation == "":
        # Return zero tensor on the correct device
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # Find information spans in continuation (character offsets relative to continuation)
    info_spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"<information>.*?</information>", continuation, flags=re.DOTALL | re.IGNORECASE):
        info_spans.append((m.start(), m.end()))

    # Tokenize prompt and continuation separately (so offsets for continuation are relative to continuation text)
    enc_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    enc_cont = tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    prompt_ids = enc_prompt["input_ids"]
    cont_ids = enc_cont["input_ids"]
    offsets = enc_cont["offset_mapping"][0].tolist()  # list of (start, end) pairs per cont token

    # Build full input ids = prompt + continuation and move to device
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1).to(device)
    
    cont_start = prompt_ids.shape[1]
    seq_len = input_ids.shape[1]

    # Forward pass (keep grad)
    outputs = model(input_ids, return_dict=True)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Positions predicting continuation tokens:
    # predictions at positions cont_start .. seq_len-1 are predicted by logits at cont_start-1 .. seq_len-2
    if seq_len - cont_start <= 0:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # Create position tensors and move to device
    pred_pos = torch.arange(cont_start - 1, seq_len - 1, device=device)
    target_pos = torch.arange(cont_start, seq_len, device=device)

    logits_preds = logits[0, pred_pos, :]  # shape (num_cont_tokens, vocab)
    target_ids = input_ids[0, target_pos]  # shape (num_cont_tokens,)

    log_probs = F.log_softmax(logits_preds, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # (num_cont_tokens,)

    # Build mask: True for tokens to KEEP (not in <information>), False to drop
    keep_mask = torch.ones_like(token_log_probs, dtype=torch.bool)
    for i, (tok_start, tok_end) in enumerate(offsets):
        # if token overlaps any info_span -> drop it
        for (inf_s, inf_e) in info_spans:
            # overlap if not (tok_end <= inf_s or tok_start >= inf_e)
            if not (tok_end <= inf_s or tok_start >= inf_e):
                keep_mask[i] = False
                break

    # Sum log-probs over kept tokens
    if keep_mask.sum().item() == 0:
        # no tokens to train on in this continuation
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    kept_log_probs = token_log_probs[keep_mask]
    return kept_log_probs.sum()


# -----------------------------
# Efficient batch trajectory generation
# -----------------------------
def generate_batch_trajectories(
    system,  # InferenceSystem type
    questions: List[Dict[str, Any]],
    sample_trajectories: int = 3,
    max_turns: int = 5
) -> List[List[Dict[str, Any]]]:
    """
    Generate multiple trajectories for a batch of questions using the InferenceSystem.
    This is highly optimized: creates all question copies at once and calls inference() once.
    
    Args:
        system: The InferenceSystem instance
        questions: List of question data with 'id', 'question', 'golden_answers'
        sample_trajectories: Number of trajectories to generate per question
        max_turns: Maximum number of turns per trajectory
        
    Returns:
        List of lists, where each inner list contains trajectories for one question
    """
    logger.info(f"Starting batch trajectory generation for {len(questions)} questions")
    # Create all question copies at once for efficient batch processing
    all_question_copies = []
    question_to_trajectory_mapping = []  # Maps result index back to (original_question_idx, trajectory_idx)
    
    logger.info("Preparing question copies for batch processing...")
    for q_idx, question_data in enumerate(questions):
        for traj_idx in range(sample_trajectories):
            # Create a copy of the question with a unique ID for this trajectory
            question_copy = question_data.copy()
            question_copy["id"] = f"{question_data['id']}_traj_{traj_idx}"
            question_copy["original_id"] = question_data["id"]  # Keep track of original ID
            question_copy["trajectory_idx"] = traj_idx
            
            all_question_copies.append(question_copy)
            question_to_trajectory_mapping.append((q_idx, traj_idx))
    
    logger.info(f"Generating {len(all_question_copies)} trajectories ({sample_trajectories} per question) in single batch")
    
    # Single call to inference for all trajectories
    try:
        logger.info("Calling system.inference for batch generation...")
        all_results = system.inference(all_question_copies, max_turns=max_turns)
        logger.info(f"Generated {len(all_results)} results from {len(all_question_copies)} input questions")
        
        # Debug: Log some result IDs to understand the format
        if all_results:
            sample_ids = [result.get("id", "no_id") for result in all_results[:5]]
            logger.info(f"Sample result IDs: {sample_ids}")
    except Exception as e:
        logger.error(f"Error in batch inference: {e}")
        # Return empty trajectories for all questions
        return [[] for _ in questions]
    
    # Group results back by original question
    all_trajectories = [[] for _ in questions]
    mapped_count = 0
    
    for result in all_results:
        # Extract original question index and trajectory index from the result ID
        result_id = result.get("id", "")
        
        # Parse the result ID to extract original_id and trajectory_idx
        # Expected format: "original_id_traj_trajectory_idx"
        if "_traj_" in result_id:
            try:
                # Split by "_traj_" and extract parts
                parts = result_id.split("_traj_")
                if len(parts) == 2:
                    original_id = parts[0]
                    trajectory_idx = int(parts[1])
                    
                    # Find the original question index
                    for q_idx, question_data in enumerate(questions):
                        if question_data["id"] == original_id:
                            # Add trajectory index back to result for sorting
                            result["trajectory_idx"] = trajectory_idx
                            all_trajectories[q_idx].append(result)
                            mapped_count += 1
                            break
                    else:
                        logger.warning(f"Could not find original question for ID: {original_id}")
                else:
                    logger.warning(f"Invalid trajectory ID format: {result_id}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse trajectory ID {result_id}: {e}")
        else:
            logger.warning(f"Result ID does not contain trajectory pattern: {result_id}")
    
    logger.info(f"Successfully mapped {mapped_count} out of {len(all_results)} results")
    
    # Ensure all questions have the right number of trajectories (fill with None if missing)
    for q_idx, question_trajectories in enumerate(all_trajectories):
        while len(question_trajectories) < sample_trajectories:
            logger.warning(f"Missing trajectory for question {questions[q_idx]['id']}, adding None")
            question_trajectories.append(None)
        
        # Sort by trajectory index to ensure consistent order
        question_trajectories.sort(key=lambda x: x.get("trajectory_idx", -1) if x is not None else -1)
    
    logger.info("Batch trajectory generation completed successfully")
    return all_trajectories


def extract_trajectory_text(trajectory_result: Dict[str, Any]) -> str:
    """
    Extract the full trajectory text from a trajectory result.
    
    Args:
        trajectory_result: Result from system.inference()
        
    Returns:
        Full trajectory text (response field)
    """
    if trajectory_result is None:
        return ""
    
    return trajectory_result.get("response", "")


# -----------------------------
# DPO training routine
# -----------------------------
class DPOTrainingDataset(Dataset):
    def __init__(self, items: List[dict]):
        # Expect each item to have "prompt" and "reference_trajectory"
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def train_dpo(
    data_path: str,
    base_model_path: str,
    lora_adapter_path: Optional[str],
    summarizer_model_name: str,
    retriever_type: str,
    retriever_index_path: str,
    e5_model_path: str,
    device: str,
    output_dir: str,
    epochs: int = 1,
    sample_trajectories: int = 3,
    batch_size: int = 10,
    lr: float = 1e-4,
    tau: float = 1.0,
    grad_accum_steps: int = 1,
    save_every_steps: int = 200,
    max_data_samples: int = None,
    high_randomness: bool = True,
):
    logger.info("=" * 70)
    logger.info("DPO Training Configuration")
    logger.info("=" * 70)
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Base Model: {base_model_path}")
    logger.info(f"LoRA Adapter: {lora_adapter_path}")
    logger.info(f"Summarizer Model: {summarizer_model_name}")
    logger.info(f"Retriever Type: {retriever_type}")
    logger.info(f"Retriever Index: {retriever_index_path}")
    logger.info(f"E5 Model Path: {e5_model_path}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Sample Trajectories: {sample_trajectories}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {lr}")
    logger.info(f"Tau: {tau}")
    logger.info(f"Output Dir: {output_dir}")
    logger.info(f"Max Data Samples: {max_data_samples if max_data_samples else 'All'}")
    logger.info(f"High Randomness Mode: {high_randomness}")
    logger.info("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)

    device_t = torch.device(device)
    logger.info(f"Using manual device management. Device: {device_t}")

    # Load dataset
    logger.info(f"Loading dataset from {data_path}...")
    items = load_dataset(data_path, max_data_samples)

    # Normalize and validate entries
    normalized = []
    id = 0
    for it in items:
        # Extract required fields
        qid = it.get("id")
        if not qid:
            qid = f"dpo_{id}"
            id += 1
        prompt = it.get("instruction") or it.get("prompt")
        input_text = it.get("input")
        ref_traj = it.get("reference_trajectory") or it.get("ref") or it.get("reference") or it.get("output")
        
        # Validate required fields
        if not qid:
            logger.warning(f"Skipping item missing id: {it}")
            continue
        if not prompt:
            logger.warning(f"Skipping item {qid} missing prompt: {it}")
            continue
        if not input_text:
            logger.warning(f"Skipping item {qid} missing input: {it}")
            continue
        
        # Build conversation from prompt and input (similar to SFT format)
        if input_text:
            # Use input as the main question, prompt as context
            full_question = f"{prompt}\n\n{input_text}"
        else:
            # Fallback to just prompt if no input
            full_question = prompt
        
        if not ref_traj:
            logger.warning(f"Item {qid} missing reference_trajectory; still kept (ref empty).")
            ref_traj = ""
        
        normalized.append({
            "id": qid, 
            "prompt": full_question,  # Combined prompt and input
            "question": full_question,  # For inference method
            "reference_trajectory": ref_traj,
            "golden_answers": [ref_traj] if ref_traj else []  # For inference method
        })

    # Create an InferenceConfig for InferenceSystem to use for generation (so its internal prompts/templates match main.py)
    # For DPO, we use the LoRA-adapted model for generation to ensure consistency with training
    # Use more aggressive random settings for trajectory generation to create diverse samples
    inf_cfg = InferenceConfig(
        reasoner_model_name=base_model_path,  # Use base model path
        summarizer_model_name=summarizer_model_name,
        reasoner_lora_path=lora_adapter_path,  # Use LoRA adapter for generation
        summarizer_lora_path=None,
        retriever_type=retriever_type,
        retriever_index_path=retriever_index_path,
        e5_model_path=e5_model_path,  # Use specified E5 model path
        # Enable high randomness for diverse trajectory generation
        greedy_thinking=False,  # Use sampling instead of greedy
        high_randomness_mode=high_randomness,  # Use parameter for aggressive random generation
    )

    # Initialize InferenceSystem for generation (this will load all components used exclusively for generation)
    logger.info("Initializing InferenceSystem (for generation of trajectories)...")
    logger.info("  - Loading reasoner model...")
    logger.info("  - Loading summarizer model...")
    logger.info("  - Loading retriever...")
    if high_randomness:
        logger.info("  - High randomness mode enabled for diverse trajectory generation")
    else:
        logger.info("  - Standard randomness mode for trajectory generation")
    system = InferenceSystem(inf_cfg)
    tokenizer = system.reasoner.tokenizer
    logger.info("InferenceSystem initialized successfully")

    # Prepare base model + apply LoRA adapter (for training)
    logger.info(f"Loading base model for training from {base_model_path} ...")
    # Use device_map="auto" for multi-GPU support
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model = base_model
    if lora_adapter_path and os.path.isdir(lora_adapter_path):
        logger.info(f"Applying LoRA adapter from {lora_adapter_path} (PEFT)...")
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
    else:
        logger.info("No LoRA adapter specified or not found. Training will fine-tune the base model (may be very large).")

    model.train()
    # Prepare optimizer for trainable params only (PEFT ensures only adapter params require grad)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = AdamW(trainable, lr=lr)
    
    # Prepare model and optimizer (without Accelerate)
    logger.info("Preparing model and optimizer...")
    # Move model to device if not already there
    try:
        model = model.to(device_t)
        logger.info(f"Model moved to {device_t}")
    except Exception as e:
        logger.warning(f"Could not move model to {device_t}: {e}. Model may already be on correct device(s).")
    logger.info("Model and optimizer prepared successfully")

    global_step = 0
    
    # Create batches for DPO training (not for trajectory generation)
    def create_training_batches(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    # Create overall epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="DPO Training Epochs", unit="epoch", position=0)
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"==== Starting epoch {epoch+1}/{epochs} ====")
        logger.info(f"Processing {len(normalized)} questions")
        
        # Phase 1: Generate ALL trajectories for ALL questions at once (maximum efficiency)
        logger.info("Phase 1: Generating all trajectories in single batch...")
        logger.info(f"  - Generating {sample_trajectories} trajectories per question")
        logger.info(f"  - Total trajectories to generate: {len(normalized) * sample_trajectories}")
        
        logger.info("Starting trajectory generation...")
        # Add progress bar for trajectory generation
        traj_pbar = tqdm(total=1, desc=f"Epoch {epoch+1}: Generating {len(normalized) * sample_trajectories} trajectories", unit="batch", position=1)
        all_trajectories = generate_batch_trajectories(
            system, normalized, sample_trajectories, max_turns=5
        )
        traj_pbar.update(1)
        traj_pbar.set_postfix({
            'questions': len(normalized),
            'trajectories_per_q': sample_trajectories,
            'total_trajectories': len(normalized) * sample_trajectories
        })
        traj_pbar.close()
        logger.info("Trajectory generation completed")
        
        # Log trajectory generation results
        total_trajectories = sum(len(trajs) for trajs in all_trajectories)
        valid_trajectories = sum(len([t for t in trajs if t is not None]) for trajs in all_trajectories)
        logger.info(f"Phase 1 completed: Generated {valid_trajectories}/{total_trajectories} valid trajectories")
        
        # Save detailed trajectory information for debugging
        trajectory_debug_info = []
        for q_idx, question_data in enumerate(normalized):
            question_info = {
                "question_id": question_data["id"],
                "question": question_data["question"],
                "golden_answers": question_data["golden_answers"],
                "trajectories": []
            }
            
            if q_idx < len(all_trajectories):
                for traj_idx, trajectory in enumerate(all_trajectories[q_idx]):
                    if trajectory is not None:
                        traj_text = extract_trajectory_text(trajectory)
                        # Check if trajectory is correct using the same logic as training loop
                        traj_answer = extract_answer_from_trajectory(traj_text)
                        # Check against all golden answers
                        is_correct = any(
                            is_exact_match(traj_answer, extract_answer_from_trajectory(golden))
                            for golden in question_data["golden_answers"]
                        )
                        
                        traj_info = {
                            "trajectory_index": traj_idx,
                            "trajectory_id": trajectory.get("id", f"unknown_{traj_idx}"),
                            "is_correct": is_correct,
                            "trajectory_text": traj_text,
                            "source": "generated"
                        }
                    else:
                        traj_info = {
                            "trajectory_index": traj_idx,
                            "trajectory_id": f"{question_data['id']}_traj_{traj_idx}",
                            "is_correct": False,
                            "trajectory_text": None,
                            "source": "missing"
                        }
                    question_info["trajectories"].append(traj_info)
            
            trajectory_debug_info.append(question_info)
        
        # Save all generated trajectories
        all_trajectories_file = os.path.join(output_dir, f"all_generated_trajectories_epoch_{epoch+1}.json")
        with open(all_trajectories_file, "w", encoding="utf-8") as f:
            json.dump(trajectory_debug_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved all generated trajectories to {all_trajectories_file}")
        
        # Clear GPU cache after trajectory generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared after trajectory generation")
        
        # Phase 2: DPO training in smaller batches for memory efficiency
        logger.info(f"Phase 2: DPO training in batches of {batch_size} with gradient accumulation of {grad_accum_steps}...")
        
        # Create training batches from the normalized data and their trajectories
        training_data = list(zip(normalized, all_trajectories))
        total_batches = (len(training_data) + batch_size - 1) // batch_size
        logger.info(f"  - Total training batches: {total_batches}")
        logger.info(f"  - Effective batch size: {batch_size * grad_accum_steps}")
        
        # Collect all DPO trajectories used for training
        all_dpo_trajectories = []
        
        # Create progress bar for DPO training batches
        training_batches = list(create_training_batches(training_data, batch_size))
        for batch_idx, training_batch in enumerate(tqdm(training_batches, desc=f"Epoch {epoch+1} DPO Training", unit="batch", position=1)):
            logger.info(f"Processing DPO batch {batch_idx + 1}/{total_batches} with {len(training_batch)} questions")
            
            batch_losses = []
            valid_pairs = 0
            skipped_questions = 0
            batch_dpo_pairs = []  # Track DPO pairs for this batch
            
            # Add progress bar for individual questions within batch
            question_pbar = tqdm(training_batch, desc=f"Batch {batch_idx + 1}", unit="question", position=2, leave=False)
            for question_idx, (question_data, question_trajectories) in enumerate(question_pbar):
                qid = question_data.get("id")
                prompt = question_data["prompt"]
                reference_traj = question_data["reference_trajectory"]
                
                # Convert trajectory results to text
                trajectories = [extract_trajectory_text(traj) for traj in question_trajectories]
                
                # Extract answers from each trajectory and from the reference trajectory
                traj_answers = [extract_answer_from_trajectory(t) for t in trajectories]
                ref_answer = extract_answer_from_trajectory(reference_traj)
                
                # Decide correctness by comparing extracted answers to reference answer (exact-match normalized)
                correctness = []
                for ans in traj_answers:
                    if ref_answer:
                        correctness.append(is_exact_match(ans, ref_answer))
                    else:
                        # if no ref answer we mark as False (so that if all wrong we skip)
                        correctness.append(False)
                
                n_correct = sum(correctness)
                n_total = len(correctness)
                
                # DPO pairing logic:
                pos_traj = None
                neg_traj = None
                reason = None
                
                # Track DPO pair information for debugging
                dpo_pair_info = {
                    "question_id": qid,
                    "positive_trajectory_id": None,
                    "negative_trajectory_id": None,
                    "positive_source": None,
                    "negative_source": None,
                    "positive_text": None,
                    "negative_text": None,
                    "pairing_reason": None
                }
                
                # Case: at least one correct AND at least one wrong
                if n_correct >= 1 and (n_total - n_correct) >= 1:
                    pos_idx = correctness.index(True)
                    neg_idx = correctness.index(False)
                    pos_traj = trajectories[pos_idx]
                    neg_traj = trajectories[neg_idx]
                    reason = "mixed"
                    valid_pairs += 1
                    
                    # Track pair info
                    dpo_pair_info["positive_trajectory_id"] = pos_traj.get("id", f"{qid}_traj_{pos_idx}") if isinstance(pos_traj, dict) else f"{qid}_traj_{pos_idx}"
                    dpo_pair_info["negative_trajectory_id"] = neg_traj.get("id", f"{qid}_traj_{neg_idx}") if isinstance(neg_traj, dict) else f"{qid}_traj_{neg_idx}"
                    dpo_pair_info["positive_source"] = "generated_correct"
                    dpo_pair_info["negative_source"] = "generated_incorrect"
                    dpo_pair_info["positive_text"] = extract_trajectory_text(pos_traj) if isinstance(pos_traj, dict) else pos_traj
                    dpo_pair_info["negative_text"] = extract_trajectory_text(neg_traj) if isinstance(neg_traj, dict) else neg_traj
                    dpo_pair_info["pairing_reason"] = reason
                    
                    logger.debug(f"qid {qid}: Mixed results ({n_correct}/{n_total} correct) - using trajectory {pos_idx} as positive, {neg_idx} as negative")
                elif n_correct == 0:
                    # all wrong -> use reference_traj as positive if present, otherwise skip
                    if reference_traj and reference_traj.strip():
                        pos_traj = reference_traj
                        neg_traj = trajectories[0] if trajectories else ""
                        reason = "all_wrong_use_reference_pos"
                        valid_pairs += 1
                        
                        # Track pair info
                        dpo_pair_info["positive_trajectory_id"] = f"{qid}_reference"
                        dpo_pair_info["negative_trajectory_id"] = trajectories[0].get("id", f"{qid}_traj_0") if trajectories and isinstance(trajectories[0], dict) else f"{qid}_traj_0" if trajectories else "none"
                        dpo_pair_info["positive_source"] = "reference"
                        dpo_pair_info["negative_source"] = "generated_incorrect"
                        dpo_pair_info["positive_text"] = reference_traj
                        dpo_pair_info["negative_text"] = extract_trajectory_text(trajectories[0]) if trajectories and isinstance(trajectories[0], dict) else trajectories[0] if trajectories else "none"
                        dpo_pair_info["pairing_reason"] = reason
                        
                        logger.debug(f"qid {qid}: All trajectories wrong ({n_correct}/{n_total}) - using reference as positive")
                    else:
                        logger.info(f"Skipping qid {qid}: all generated trajectories wrong and no reference trajectory available.")
                        skipped_questions += 1
                        continue
                else:
                    # all correct => skip
                    logger.info(f"Skipping qid {qid}: all {n_total} trajectories correct.")
                    skipped_questions += 1
                    continue
                
                # Store DPO pair info for this batch
                batch_dpo_pairs.append(dpo_pair_info)
                
                # Add to overall DPO trajectories collection
                all_dpo_trajectories.append(dpo_pair_info)
                
                # Compute masked log-probs for pos and neg continuations
                try:
                    # Build the initial sequence for log-prob calculation (same as used in trajectory generation)
                    prompted_question = system.reasoner.prompt_template.format(question=prompt)
                    messages = [{"role": "user", "content": prompted_question}]
                    initial_sequence = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    
                    # Note: pos_traj and neg_traj are the full generated responses (including search actions, information blocks, etc.)
                    # We need to compute log-probs for these continuations given the initial sequence
                    # Accelerate will handle device placement automatically
                    logp_pos = compute_masked_logprob(model, tokenizer, initial_sequence, pos_traj, device_t)
                    logp_neg = compute_masked_logprob(model, tokenizer, initial_sequence, neg_traj, device_t)
                    
                    # DPO loss: stable form (softplus)
                    diff = (logp_pos - logp_neg) / (tau if tau > 0 else 1.0)
                    loss = F.softplus(-diff)  # equals -log sigmoid(diff) in a stable way
                    
                    batch_losses.append(loss)
                    global_step += 1
                    
                    # Update progress bar description with current loss
                    question_pbar.set_postfix({
                        'qid': qid[:20] + '...' if len(qid) > 20 else qid,
                        'loss': f"{loss.item():.4f}",
                        'reason': reason[:15] + '...' if len(str(reason)) > 15 else str(reason),
                        'step': global_step
                    })
                    
                    if global_step % 10 == 0:
                        logger.info(f"Step {global_step} | qid {qid} | loss {loss.item():.4f} | reason {reason}")
                    else:
                        logger.debug(f"qid {qid} | loss {loss.item():.4f} | reason {reason}")
                
                except Exception as e:
                    logger.error(f"Error computing DPO loss for qid {qid}: {e}")
                    continue
            
            # Log batch statistics
            logger.info(f"Batch {batch_idx + 1} completed: {valid_pairs} valid DPO pairs, {skipped_questions} skipped questions")
            
            # Phase 3: Backpropagate all losses in this batch
            if batch_losses:
                # Add progress bar for backpropagation
                with tqdm(total=3, desc=f"Backprop Batch {batch_idx + 1}", unit="step", position=3, leave=False) as bp_pbar:
                    logger.info(f"Backpropagating {len(batch_losses)} losses in batch...")
                    bp_pbar.update(1)
                    
                    optimizer.zero_grad()
                    bp_pbar.update(1)
                    
                    # Sum all losses in the batch
                    total_loss = sum(batch_losses)
                    total_loss.backward()
                    optimizer.step()
                    bp_pbar.update(1)
                    
                    bp_pbar.set_postfix({
                        'avg_loss': f"{total_loss.item() / len(batch_losses):.4f}",
                        'total_loss': f"{total_loss.item():.4f}",
                        'pairs': len(batch_losses)
                    })
                
                logger.info(f"Batch {batch_idx + 1} completed. Average loss: {total_loss.item() / len(batch_losses):.4f}")
                
                # Periodic save
                if global_step % save_every_steps == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    try:
                        # Save model directly (no unwrapping needed without Accelerate)
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        logger.info(f"Saved checkpoint to {ckpt_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint at step {global_step}: {e}")
            else:
                logger.info(f"Batch {batch_idx + 1} skipped - no valid DPO pairs found")
            
            # Save DPO pairs information for this batch
            if batch_dpo_pairs:
                dpo_pairs_file = os.path.join(output_dir, f"dpo_pairs_batch_{batch_idx + 1}_epoch_{epoch+1}.json")
                with open(dpo_pairs_file, "w", encoding="utf-8") as f:
                    json.dump(batch_dpo_pairs, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved DPO pairs info for batch {batch_idx + 1} to {dpo_pairs_file}")
        
        logger.info(f"Epoch {epoch+1} completed! Total steps: {global_step}")
        
        # Save all DPO trajectories used for training
        dpo_trajectories_file = os.path.join(output_dir, f"dpo_trajectories_used_epoch_{epoch+1}.json")
        with open(dpo_trajectories_file, "w", encoding="utf-8") as f:
            json.dump(all_dpo_trajectories, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved DPO trajectories used for training to {dpo_trajectories_file}")
        
        # Save DPO training summary
        dpo_summary = {
            "epoch": epoch + 1,
            "total_questions": len(normalized),
            "total_trajectories_generated": total_trajectories,
            "valid_trajectories": valid_trajectories,
            "total_training_steps": global_step,
            "dpo_pairs_used": len(all_dpo_trajectories),
            "training_summary": "See all_generated_trajectories_epoch_{}.json for all trajectories and dpo_trajectories_used_epoch_{}.json for DPO training pairs".format(epoch + 1, epoch + 1)
        }
        
        summary_file = os.path.join(output_dir, f"dpo_training_summary_epoch_{epoch+1}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(dpo_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved DPO training summary to {summary_file}")

    # Final save
    final_dir = os.path.join(output_dir, "dpo_final")
    os.makedirs(final_dir, exist_ok=True)
    try:
        # Save model directly (no unwrapping needed without Accelerate)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved final model/adapters to {final_dir}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="DPO training for agentic RAG reasoner (uses main.Reasoner for generation)")
    parser.add_argument("--data", type=str, default="dpo_training_data.json", help="Path to training data (JSON or JSONL)")
    parser.add_argument("--base-model", type=str, default="/scratch/yl9038/models/Qwen3-0.6B", help="Path to base causal LM model")
    parser.add_argument("--lora-adapter", type=str, default="models/llamafactory/qwen3-lora-0.6B", help="Path to LoRA adapter (optional)")
    parser.add_argument("--summarizer-model", type=str, default="/scratch/yl9038/models/Qwen3-32B", help="Summarizer model path (used only to build InferenceConfig)")
    parser.add_argument("--retriever-type", type=str, default="e5", choices=["e5", "bm25"], help="Retriever type for InferenceConfig used by Reasoner")
    parser.add_argument("--retriever-index", type=str, default="indexes/wiki2021-e5", help="Retriever index path (used by Reasoner InferenceConfig)")
    parser.add_argument("--e5-model-path", type=str, default="/scratch/yl9038/models/e5-large-v2", help="E5 model path (used by E5 retriever)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu)")
    parser.add_argument("--output-dir", type=str, default="dpo_output", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sample-trajectories", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10, help="Number of questions to process in each batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--max-data-samples", type=int, default=None, help="Maximum number of data samples to use (None for all)")
    parser.add_argument("--high-randomness", action="store_true", default=True, help="Enable high randomness mode for diverse trajectory generation (default: True)")
    parser.add_argument("--no-high-randomness", dest="high_randomness", action="store_false", help="Disable high randomness mode")
    args = parser.parse_args()

    train_dpo(
        data_path=args.data,
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        summarizer_model_name=args.summarizer_model,
        retriever_type=args.retriever_type,
        retriever_index_path=args.retriever_index,
        e5_model_path=args.e5_model_path,
        device=args.device,
        output_dir=args.output_dir,
        epochs=args.epochs,
        sample_trajectories=args.sample_trajectories,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        save_every_steps=args.save_every_steps,
        max_data_samples=args.max_data_samples,
        high_randomness=args.high_randomness,
    )


if __name__ == "__main__":
    main()
