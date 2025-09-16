#!/usr/bin/env python3
"""
train_dpo.py

ON-POLICY DPO training script that:
- uses Reasoner from main.py to sample trajectories (so generation matches inference)
- processes each question sequentially: generate trajectories → train immediately → move to next question
- masks tokens inside <information>...</information> in continuations when computing log-probs
- constructs (pos, neg) pairs per your rules and trains via DPO objective
- updates model after each question (true on-policy training)

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
from tqdm import tqdm

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import PeftModel

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


# -----------------------------
# Device management utilities
# -----------------------------
def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device of the model's first parameter.
    This is useful for multi-GPU models where different parameters might be on different devices.
    """
    return next(model.parameters()).device

def ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Ensure a tensor is on the specified device.
    Returns the tensor moved to the device if needed.
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor

# -----------------------------
# Masked log-prob calculation
# -----------------------------
def compute_masked_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    return_length: bool = False,
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
        # Return zero tensor on the specified device
        if return_length:
            return torch.tensor(0.0, device=device, dtype=torch.float32), 0
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # Find information spans in continuation (character offsets relative to continuation)
    info_spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"<information>.*?</information>", continuation, flags=re.DOTALL | re.IGNORECASE):
        info_spans.append((m.start(), m.end()))
    
    # Debug: log information spans found
    if info_spans:
        logger.debug(f"Found {len(info_spans)} information spans in continuation of length {len(continuation)}")
        for i, (start, end) in enumerate(info_spans):
            logger.debug(f"Info span {i}: chars {start}-{end}, content preview: {continuation[start:min(start+50, end)]}...")

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

    # Build full input ids = prompt + continuation and move to the specified device
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1).to(device)
    
    cont_start = prompt_ids.shape[1]
    seq_len = input_ids.shape[1]

    # Forward pass (keep grad)
    outputs = model(input_ids, return_dict=True)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Positions predicting continuation tokens:
    # predictions at positions cont_start .. seq_len-1 are predicted by logits at cont_start-1 .. seq_len-2
    if seq_len - cont_start <= 0:
        if return_length:
            return torch.tensor(0.0, device=device, dtype=torch.float32), 0
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    # Create position tensors directly on the specified device to avoid device mismatch
    pred_pos = torch.arange(cont_start - 1, seq_len - 1, device=device, dtype=torch.long)
    target_pos = torch.arange(cont_start, seq_len, device=device, dtype=torch.long)

    logits_preds = logits[0, pred_pos, :]  # shape (num_cont_tokens, vocab)
    target_ids = input_ids[0, target_pos]  # shape (num_cont_tokens,)

    log_probs = F.log_softmax(logits_preds, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # (num_cont_tokens,)

    # Build mask: True for tokens to KEEP (not in <information>), False to drop
    # Ensure mask is on the same device as token_log_probs
    keep_mask = torch.ones_like(token_log_probs, dtype=torch.bool, device=token_log_probs.device)
    masked_tokens = 0
    for i, (tok_start, tok_end) in enumerate(offsets):
        # if token overlaps any info_span -> drop it
        for (inf_s, inf_e) in info_spans:
            # overlap if not (tok_end <= inf_s or tok_start >= inf_e)
            if not (tok_end <= inf_s or tok_start >= inf_e):
                keep_mask[i] = False
                masked_tokens += 1
                break
    
    # Debug: log masking statistics
    total_tokens = len(keep_mask)
    kept_tokens = keep_mask.sum().item()
    logger.debug(f"Masking: {masked_tokens}/{total_tokens} tokens masked, {kept_tokens}/{total_tokens} tokens kept")

    # Sum log-probs over kept tokens
    if keep_mask.sum().item() == 0:
        # no tokens to train on in this continuation
        logger.warning(f"All tokens masked out in continuation. Total tokens: {len(keep_mask)}, Info spans: {len(info_spans)}")
        
        # Debug: show what content remains after removing information blocks
        remaining_content = continuation
        for (start, end) in reversed(info_spans):  # Reverse to maintain indices
            remaining_content = remaining_content[:start] + remaining_content[end:]
        logger.warning(f"Content after removing info blocks: '{remaining_content.strip()}'")
        
        if return_length:
            return torch.tensor(0.0, device=token_log_probs.device, dtype=torch.float32), 0
        return torch.tensor(0.0, device=token_log_probs.device, dtype=torch.float32)

    kept_log_probs = token_log_probs[keep_mask]
    total_log_prob = kept_log_probs.sum()
    num_kept_tokens = keep_mask.sum().item()
    
    # Debug logging for very small or zero log probabilities
    if abs(total_log_prob.item()) < 1e-6:
        logger.debug(f"Very small log probability: {total_log_prob.item():.8f} | Kept tokens: {keep_mask.sum().item()}/{len(keep_mask)} | Info spans: {len(info_spans)}")
        
        # Show some examples of kept vs masked tokens
        kept_token_indices = torch.where(keep_mask)[0][:10]  # First 10 kept tokens
        masked_token_indices = torch.where(~keep_mask)[0][:10]  # First 10 masked tokens
        
        if len(kept_token_indices) > 0:
            kept_tokens_text = [tokenizer.decode([cont_ids[0][idx]]) for idx in kept_token_indices]
            logger.debug(f"Sample kept tokens: {kept_tokens_text}")
        
        if len(masked_token_indices) > 0:
            masked_tokens_text = [tokenizer.decode([cont_ids[0][idx]]) for idx in masked_token_indices]
            logger.debug(f"Sample masked tokens: {masked_tokens_text}")
    
    if return_length:
        return total_log_prob, num_kept_tokens
    return total_log_prob


def test_masking_logic():
    """
    Test function to verify that masking logic handles multiple information blocks correctly.
    This can be called for debugging purposes.
    """
    test_continuation = """
    <thinking>
    I need to solve this problem step by step.
    </thinking>
    
    <information>
    Some external knowledge here.
    </information>
    
    <thinking>
    Based on the information, I can proceed.
    </thinking>
    
    <information>
    More external knowledge here.
    </information>
    
    <answer>
    The final answer is 42.
    </answer>
    """
    
    # Test regex to find information spans
    info_spans = []
    for m in re.finditer(r"<information>.*?</information>", test_continuation, flags=re.DOTALL | re.IGNORECASE):
        info_spans.append((m.start(), m.end()))
    
    print(f"Found {len(info_spans)} information spans:")
    for i, (start, end) in enumerate(info_spans):
        print(f"  Span {i}: chars {start}-{end}")
        print(f"  Content: {test_continuation[start:end][:100]}...")
    
    # Test what remains after removing info blocks
    remaining_content = test_continuation
    for (start, end) in reversed(info_spans):
        remaining_content = remaining_content[:start] + remaining_content[end:]
    
    print(f"\nContent after removing info blocks:")
    print(f"'{remaining_content.strip()}'")
    
    return info_spans, remaining_content


# -----------------------------
# On-policy trajectory generation (per question)
# -----------------------------
def generate_trajectories_for_question(
    system,  # InferenceSystem type
    question_data: Dict[str, Any],
    sample_trajectories: int = 3,
    max_turns: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate multiple trajectories for a single question using the InferenceSystem.
    This is on-policy: generates trajectories using the current model state.
    
    Args:
        system: The InferenceSystem instance
        question_data: Question data with 'id', 'question', 'golden_answers'
        sample_trajectories: Number of trajectories to generate per question
        max_turns: Maximum number of turns per trajectory
        
    Returns:
        List of trajectory results for this question
    """
    logger.debug(f"Generating {sample_trajectories} trajectories for question {question_data['id']}")
    
    # Create question copies for this single question
    question_copies = []
    for traj_idx in range(sample_trajectories):
        # Create a copy of the question with a unique ID for this trajectory
        question_copy = question_data.copy()
        question_copy["id"] = f"{question_data['id']}_traj_{traj_idx}"
        question_copy["original_id"] = question_data["id"]  # Keep track of original ID
        question_copy["trajectory_idx"] = traj_idx
        
        question_copies.append(question_copy)
    
    # Generate trajectories for this question
    try:
        logger.debug(f"Calling system.inference for question {question_data['id']}...")
        results = system.inference(question_copies, max_turns=max_turns)
        logger.debug(f"Generated {len(results)} results for question {question_data['id']}")
        
        # Sort results by trajectory index to ensure consistent order
        results.sort(key=lambda x: x.get("trajectory_idx", -1))
        
        # Ensure we have the right number of trajectories (fill with None if missing)
        while len(results) < sample_trajectories:
            logger.warning(f"Missing trajectory for question {question_data['id']}, adding None")
            results.append(None)
        
        return results
        
    except Exception as e:
        logger.error(f"Error generating trajectories for question {question_data['id']}: {e}")
        # Return empty trajectories for this question
        return [None] * sample_trajectories


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
    lr: float = 1e-4,
    tau: float = 1.0,
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
    
    # For multi-GPU setups, we'll use a single device approach to avoid device_map issues
    # This ensures all model parameters are on the same device for consistent training
    if torch.cuda.device_count() > 1:
        logger.info(f"Multiple CUDA devices detected ({torch.cuda.device_count()}). Using single device approach for training.")
        logger.info("Note: For true multi-GPU training, consider using DistributedDataParallel or similar.")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
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
    
    # Move model to the specified device
    logger.info(f"Moving model to {device_t}...")
    model = model.to(device_t)
    logger.info(f"Model moved to {device_t}")
    
    # Prepare optimizer for trainable params only (PEFT ensures only adapter params require grad)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = AdamW(trainable, lr=lr)
    
    # Verify model is on the correct device
    model_device = get_model_device(model)
    logger.info(f"Model device: {model_device}")
    if torch.cuda.is_available():
        logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info("Model and optimizer prepared successfully")

    global_step = 0

    # Create overall epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="DPO Training Epochs", unit="epoch", position=0)
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"==== Starting epoch {epoch+1}/{epochs} ====")
        logger.info(f"Processing {len(normalized)} questions")
        logger.info("ON-POLICY TRAINING: Each question will be processed sequentially")
        
        # Collect all DPO trajectories used for training
        all_dpo_trajectories = []
        total_trajectories_generated = 0
        valid_trajectories_generated = 0
        
        # ON-POLICY TRAINING: Process each question sequentially
        question_pbar = tqdm(normalized, desc=f"Epoch {epoch+1}: On-policy DPO Training", unit="question", position=1)
        for question_idx, question_data in enumerate(question_pbar):
            qid = question_data.get("id")
            prompt = question_data["prompt"]
            reference_traj = question_data["reference_trajectory"]
            
            # Update progress bar
            question_pbar.set_postfix({
                'qid': qid[:20] + '...' if len(qid) > 20 else qid,
                'step': global_step,
                'question': f"{question_idx + 1}/{len(normalized)}"
            })
            
            logger.info(f"Processing question {question_idx + 1}/{len(normalized)}: {qid}")
            
            # Step 1: Generate trajectories for this question using current model state
            logger.debug(f"Generating {sample_trajectories} trajectories for question {qid}")
            question_trajectories = generate_trajectories_for_question(
                system, question_data, sample_trajectories, max_turns=5
            )
            
            # Update trajectory counts
            total_trajectories_generated += len(question_trajectories)
            valid_trajectories_generated += len([t for t in question_trajectories if t is not None])
            
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
                
                # Track pair info
                dpo_pair_info["positive_trajectory_id"] = f"{qid}_traj_{pos_idx}"
                dpo_pair_info["negative_trajectory_id"] = f"{qid}_traj_{neg_idx}"
                dpo_pair_info["positive_source"] = "generated_correct"
                dpo_pair_info["negative_source"] = "generated_incorrect"
                dpo_pair_info["positive_text"] = pos_traj
                dpo_pair_info["negative_text"] = neg_traj
                dpo_pair_info["pairing_reason"] = reason
                
                logger.debug(f"qid {qid}: Mixed results ({n_correct}/{n_total} correct) - using trajectory {pos_idx} as positive, {neg_idx} as negative")
            elif n_correct == 0:
                # all wrong -> use reference_traj as positive if present, otherwise skip
                if reference_traj and reference_traj.strip():
                    pos_traj = reference_traj
                    neg_traj = trajectories[0] if trajectories else ""
                    reason = "all_wrong_use_reference_pos"
                    
                    # Track pair info
                    dpo_pair_info["positive_trajectory_id"] = f"{qid}_reference"
                    dpo_pair_info["negative_trajectory_id"] = f"{qid}_traj_0" if trajectories else "none"
                    dpo_pair_info["positive_source"] = "reference"
                    dpo_pair_info["negative_source"] = "generated_incorrect"
                    dpo_pair_info["positive_text"] = reference_traj
                    dpo_pair_info["negative_text"] = trajectories[0] if trajectories else "none"
                    dpo_pair_info["pairing_reason"] = reason
                    
                    logger.debug(f"qid {qid}: All trajectories wrong ({n_correct}/{n_total}) - using reference as positive")
                else:
                    logger.info(f"Skipping qid {qid}: all generated trajectories wrong and no reference trajectory available.")
                    continue
            else:
                # all correct => skip
                logger.info(f"Skipping qid {qid}: all {n_total} trajectories correct.")
                continue
            
            # Add to overall DPO trajectories collection
            all_dpo_trajectories.append(dpo_pair_info)
            
            # Step 2: Compute DPO loss and train on this question immediately
            try:
                # Build the initial sequence for log-prob calculation (same as used in trajectory generation)
                prompted_question = system.reasoner.prompt_template.format(question=prompt)
                messages = [{"role": "user", "content": prompted_question}]
                initial_sequence = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                
                # Use the specified device for all computations to ensure consistency
                logger.debug(f"Using device: {device_t}")
                
                # Compute masked log-probs for pos and neg continuations
                logp_pos, pos_length = compute_masked_logprob(model, tokenizer, initial_sequence, pos_traj, device_t, return_length=True)
                logp_neg, neg_length = compute_masked_logprob(model, tokenizer, initial_sequence, neg_traj, device_t, return_length=True)
                
                # Normalize by sequence length to prevent bias towards longer sequences
                logger.debug(f"Before normalization: logp_pos={logp_pos.item():.6f} (length={pos_length}), logp_neg={logp_neg.item():.6f} (length={neg_length})")
                if pos_length > 0:
                    logp_pos = logp_pos / pos_length
                if neg_length > 0:
                    logp_neg = logp_neg / neg_length
                logger.debug(f"After normalization: logp_pos={logp_pos.item():.6f}, logp_neg={logp_neg.item():.6f}")
                
                # Ensure both log probabilities are on the same device
                logp_pos = ensure_tensor_on_device(logp_pos, device_t)
                logp_neg = ensure_tensor_on_device(logp_neg, device_t)
                
                # Additional safety check: verify tensors are on the same device
                if logp_pos.device != logp_neg.device:
                    logger.error(f"Device mismatch detected: logp_pos on {logp_pos.device}, logp_neg on {logp_neg.device}")
                    # Force both to the specified device
                    logp_pos = ensure_tensor_on_device(logp_pos, device_t)
                    logp_neg = ensure_tensor_on_device(logp_neg, device_t)
                
                # DPO loss: stable form (softplus)
                diff = (logp_pos - logp_neg) / (tau if tau > 0 else 1.0)
                loss = F.softplus(-diff)  # equals -log sigmoid(diff) in a stable way
                
                # Additional debugging for numerical issues
                if abs(diff.item()) < 1e-10:
                    logger.warning(f"Extremely small difference between log probabilities: {diff.item():.12e}")
                
                # Test softplus behavior for debugging
                if abs(diff.item()) < 1e-6:
                    expected_loss = F.softplus(torch.tensor(0.0, device=device_t))
                    logger.warning(f"Expected loss for diff=0: {expected_loss.item():.8f}, actual loss: {loss.item():.8f}")
                
                # Debug logging for zero loss cases
                if loss.item() < 1e-4:  # Very small loss (less than 0.0001)
                    logger.warning(f"Very small loss detected: {loss.item():.8f} | logp_pos: {logp_pos.item():.6f} | logp_neg: {logp_neg.item():.6f} | diff: {diff.item():.6f} | reason: {reason}")
                    logger.warning(f"Positive trajectory length: {len(pos_traj)} chars, Negative trajectory length: {len(neg_traj)} chars")
                    logger.warning(f"Positive trajectory preview: {pos_traj[:200]}...")
                    logger.warning(f"Negative trajectory preview: {neg_traj[:200]}...")
                    
                    # Check if both are exactly zero (completely masked)
                    if logp_pos.item() == 0.0 and logp_neg.item() == 0.0:
                        logger.error("Both trajectories have zero log probability - likely completely masked out!")
                
                # Ensure loss is on the correct device
                loss = ensure_tensor_on_device(loss, device_t)
                
                # Step 3: Backpropagate and update model immediately (on-policy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
                # Update progress bar with current loss
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
                
                # Clear GPU cache periodically to manage memory
                if global_step % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("GPU cache cleared")
            
            except Exception as e:
                logger.error(f"Error computing DPO loss for qid {qid}: {e}")
                # Log additional debugging information for device-related errors
                if "device" in str(e).lower() or "cuda" in str(e).lower():
                    logger.error(f"Device-related error detected. Model device: {get_model_device(model)}")
                    logger.error(f"Available CUDA devices: {torch.cuda.device_count()}")
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            logger.error(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
                continue
        
        logger.info(f"Epoch {epoch+1} completed! Total steps: {global_step}")
        logger.info(f"Generated {valid_trajectories_generated}/{total_trajectories_generated} valid trajectories")
        logger.info(f"Used {len(all_dpo_trajectories)} DPO pairs for training")
        
        # Save all DPO trajectories used for training
        dpo_trajectories_file = os.path.join(output_dir, f"dpo_trajectories_used_epoch_{epoch+1}.json")
        with open(dpo_trajectories_file, "w", encoding="utf-8") as f:
            json.dump(all_dpo_trajectories, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved DPO trajectories used for training to {dpo_trajectories_file}")
        
        # Save DPO training summary
        dpo_summary = {
            "epoch": epoch + 1,
            "total_questions": len(normalized),
            "total_trajectories_generated": total_trajectories_generated,
            "valid_trajectories_generated": valid_trajectories_generated,
            "total_training_steps": global_step,
            "dpo_pairs_used": len(all_dpo_trajectories),
            "training_mode": "on_policy",
            "training_summary": f"On-policy DPO training: each question processed sequentially with immediate model updates. See dpo_trajectories_used_epoch_{epoch + 1}.json for DPO training pairs."
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
    parser = argparse.ArgumentParser(description="ON-POLICY DPO training for agentic RAG reasoner (uses main.Reasoner for generation, trains on each question sequentially)")
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
        lr=args.lr,
        tau=args.tau,
        save_every_steps=args.save_every_steps,
        max_data_samples=args.max_data_samples,
        high_randomness=args.high_randomness,
    )


if __name__ == "__main__":
    main()
