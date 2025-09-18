import argparse
import json
import logging
import os
import re
import copy
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import PeftModel

from main import InferenceSystem, InferenceConfig, Reasoner, Summarizer
from utils import load_dataset, is_exact_match, extract_answer, cover_match
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # Build full input ids = prompt + continuation and move to the model's device
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    
    # For models with device_map="auto", let transformers handle device placement
    # The warning about device mismatch is just a performance warning, not an error
    
    cont_start = prompt_ids.shape[1]
    seq_len = input_ids.shape[1]

    # Forward pass (keep grad)
    outputs = model(input_ids, return_dict=True)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Create position tensors - let them be placed automatically by PyTorch
    pred_pos = torch.arange(cont_start - 1, seq_len - 1, dtype=torch.long)
    target_pos = torch.arange(cont_start, seq_len, dtype=torch.long)

    logits_preds = logits[0, pred_pos, :]  # shape (num_cont_tokens, vocab)
    target_ids = input_ids[0, target_pos].long()  # shape (num_cont_tokens,) - ensure LongTensor
    
    # Clamp logits to prevent overflow in log_softmax
    logits_preds = torch.clamp(logits_preds, min=-50, max=50)
    
    # Use F.log_softmax in float32 for numerical stability
    # Convert to float32 for the computation, then back to original dtype
    log_probs = F.log_softmax(logits_preds, dim=-1)
    
    # Convert back to original dtype if needed
    
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # (num_cont_tokens,)
    

    # Build mask: True for tokens to KEEP (not in <information>), False to drop
    keep_mask = torch.ones_like(token_log_probs, dtype=torch.bool)
    masked_tokens = 0
    for i, (tok_start, tok_end) in enumerate(offsets):
        # if token overlaps any info_span -> drop it
        for (inf_s, inf_e) in info_spans:
            # overlap if not (tok_end <= inf_s or tok_start >= inf_e)
            if not (tok_end <= inf_s or tok_start >= inf_e):
                keep_mask[i] = False
                masked_tokens += 1
                break
    
    # Sum log-probs over kept tokens
    if keep_mask.sum().item() == 0:
        # no tokens to train on in this continuation
        logger.warning(f"All tokens masked out in continuation. Total tokens: {len(keep_mask)}, Info spans: {len(info_spans)}")
        
        
        if return_length:
            return torch.tensor(0.0, dtype=torch.float32), 0
        return torch.tensor(0.0, dtype=torch.float32)

    kept_log_probs = token_log_probs[keep_mask]
    
    # Clamp log probabilities to prevent extreme values that could cause NaN in sum
    kept_log_probs = torch.clamp(kept_log_probs, min=-50, max=50)
    total_log_prob = kept_log_probs.sum()
    num_kept_tokens = keep_mask.sum().item()
    
    if return_length:
        return total_log_prob, num_kept_tokens
    return total_log_prob


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
        results = system.inference(question_copies, max_turns=max_turns)
        # Sort results by trajectory index to ensure consistent order
        results.sort(key=lambda x: x.get("trajectory_idx", -1))
        return results
        
    except Exception as e:
        logger.error(f"Error generating trajectories for question {question_data['id']}: {e}")
        # Return empty trajectories for this question
        return [None] * sample_trajectories




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
    lr: float = 5e-6,
    tau: float = 1.0,
    save_every_steps: int = 200,
    max_data_samples: int = None,
    high_randomness: bool = True,
    lora_only: bool = False,
    gradient_accumulation_steps: int = 1,
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
    logger.info(f"LoRA Only Training: {lora_only}")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logger.info("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)

    device_t = torch.device(device)

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
    
    # Multi-GPU device management
    logger.info("Configuring multi-GPU device usage...")
    if torch.cuda.device_count() > 1:
        logger.info(f"Multi-GPU environment detected: {torch.cuda.device_count()} GPUs available")
        logger.info("InferenceSystem should handle multi-GPU device management automatically")
    else:
        logger.info("Single GPU environment detected")

    # Use reasoner model as reference (already loaded and frozen)
    logger.info("Using reasoner model as reference for DPO training...")
    model_ref = system.reasoner.model
    model_ref.eval()
    
    # Create a copy of the reasoner model for training
    logger.info("Creating copy of reasoner model for training...")
    model = copy.deepcopy(system.reasoner.model)
    
    # Verify models are truly separate
    logger.info("Verifying models are separate...")
    policy_params = list(model.parameters())
    ref_params = list(model_ref.parameters())
    logger.info(f"Policy model parameters: {len(policy_params)}")
    logger.info(f"Reference model parameters: {len(ref_params)}")
    
    # Check if any parameters are shared (they shouldn't be)
    shared_count = 0
    for i, (p1, p2) in enumerate(zip(policy_params, ref_params)):
        if p1.data_ptr() == p2.data_ptr():
            shared_count += 1
            logger.warning(f"Parameter {i} is shared between policy and reference models!")
    
    if shared_count == 0:
        logger.info("✅ Models are completely separate - no shared parameters")
    else:
        logger.error(f"❌ {shared_count} parameters are shared between models!")
    
    if lora_only:
        # LoRA-only training: keep LoRA adapter separate, don't merge
        logger.info("LoRA-only training mode: keeping LoRA adapter separate")
        
        # For distributed models, avoid manual device placement
        # The model should already be properly distributed via device_map="auto"
        logger.info("Using distributed model setup - no manual device placement needed")
        
        # Set model to training mode
        model.train()
        
        # Enable only LoRA parameters for training
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'adapter' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Number of trainable params (LoRA only): {sum(p.numel() for p in trainable):,}")
        logger.info("Training LoRA adapter only")
        
    else:
        # Full model training: merge LoRA adapter into base model
        if hasattr(model, 'merge_and_unload'):
            logger.info("Merging LoRA adapter into base model parameters...")
            model = model.merge_and_unload()  # This merges LoRA weights into base model
            logger.info("LoRA adapter successfully merged into base model")
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'merge_and_unload'):
            logger.info("Merging LoRA adapter into base model parameters...")
            model = model.base_model.merge_and_unload()  # For nested PEFT models
            logger.info("LoRA adapter successfully merged into base model")
        else:
            logger.info("No LoRA adapter found to merge, using base model directly")
        
        # For distributed models, avoid manual device placement
        # The model should already be properly distributed via device_map="auto"
        logger.info("Using distributed model setup - no manual device placement needed")
        
        # Set model to training mode
        model.train()
        
        # Enable all parameters for training (full fine-tuning after LoRA merge)
        for param in model.parameters():
            param.requires_grad = True
        
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Number of trainable params (full model): {sum(p.numel() for p in trainable):,}")
        logger.info("Training full model (LoRA merged + base model parameters)")
    
    optimizer = AdamW(trainable, lr=lr)
    
    

    global_step = 0
    accumulated_loss = 0.0
    accumulation_count = 0

    # Create overall epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="DPO Training Epochs", unit="epoch", position=0)
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"==== Starting epoch {epoch+1}/{epochs} ====")
        logger.info(f"Processing {len(normalized)} questions")
        logger.info("OFF-POLICY TRAINING: Using frozen reasoner for generation, training separate model")
        
        # Collect all DPO trajectories used for training
        all_dpo_trajectories = []
        total_trajectories_generated = 0
        valid_trajectories_generated = 0
        
        # OFF-POLICY TRAINING: Process each question sequentially using frozen reasoner
        question_pbar = tqdm(normalized, desc=f"Epoch {epoch+1}: Off-policy DPO Training", unit="question", position=1)
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
            question_trajectories = generate_trajectories_for_question(
                system, question_data, sample_trajectories, max_turns=5
            )
            
            # Update trajectory counts
            total_trajectories_generated += len(question_trajectories)
            valid_trajectories_generated += len([t for t in question_trajectories if t is not None])
            
            # Extract answers from each trajectory and from the reference trajectory
            traj_answers = [extract_answer(t["response"]) for t in question_trajectories]
            ref_answer = extract_answer(reference_traj)
            
            # Decide correctness by comparing extracted answers to reference answer (exact-match normalized)
            correctness = []
            n_correct = 0
            for ans in traj_answers:
                if cover_match(ref_answer, ans) or is_exact_match(ref_answer, ans):
                    correctness.append(True)
                    n_correct += 1
                elif not ans:
                    correctness.append("No Answer")
                else:
                    # if no ref answer we mark as False (so that if all wrong we skip)
                    correctness.append(False)
            
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
                neg_idx = correctness.index("No Answer") if "No Answer" in correctness else correctness.index(False) #first solve no answer
                pos_traj = question_trajectories[pos_idx]["response"]
                neg_traj = question_trajectories[neg_idx]["response"]
                reason = "compare"
                
                # Track pair info
                dpo_pair_info["positive_trajectory_id"] = f"{qid}_traj_{pos_idx}"
                dpo_pair_info["negative_trajectory_id"] = f"{qid}_traj_{neg_idx}"
                dpo_pair_info["positive_source"] = "generated_correct"
                dpo_pair_info["negative_source"] = "generated_incorrect"
                dpo_pair_info["positive_text"] = pos_traj
                dpo_pair_info["negative_text"] = neg_traj
                dpo_pair_info["pairing_reason"] = reason
            # TEMPORARY FIX: If all trajectories are correct, artificially create a pair
            elif n_correct == n_total and n_total >= 2:
                # Use the first trajectory as positive, second as negative (artificially)
                pos_idx = 0
                neg_idx = 1
                pos_traj = question_trajectories[pos_idx]["response"]
                neg_traj = question_trajectories[neg_idx]["response"]
                reason = "artificial_pair"
                logger.info(f"🔧 ARTIFICIAL DPO PAIR for {qid}: {reason} (all trajectories were correct)")
                
                # Track pair info
                dpo_pair_info["positive_trajectory_id"] = f"{qid}_traj_{pos_idx}"
                dpo_pair_info["negative_trajectory_id"] = f"{qid}_traj_{neg_idx}"
                dpo_pair_info["positive_source"] = "generated_correct"
                dpo_pair_info["negative_source"] = "generated_incorrect"
                dpo_pair_info["positive_text"] = pos_traj
                dpo_pair_info["negative_text"] = neg_traj
                dpo_pair_info["pairing_reason"] = reason
                
            elif n_correct == 0:
                # all wrong -> use reference_traj as positive if present, otherwise skip
                if reference_traj and reference_traj.strip():
                    pos_traj = reference_traj
                    neg_idx = correctness.index("No Answer") if "No Answer" in correctness else correctness.index(False) #first solve no answer
                    neg_traj = question_trajectories[neg_idx]["response"]
                    reason = "use reference"
                    
                    # Track pair info
                    dpo_pair_info["positive_trajectory_id"] = f"{qid}_reference"
                    dpo_pair_info["negative_trajectory_id"] = f"{qid}_traj_{neg_idx}"
                    dpo_pair_info["positive_source"] = "reference"
                    dpo_pair_info["negative_source"] = "generated_incorrect"
                    dpo_pair_info["positive_text"] = reference_traj
                    dpo_pair_info["negative_text"] = neg_traj
                    dpo_pair_info["pairing_reason"] = reason
                    
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
                
                
                # Compute masked log-probs for pos and neg continuations using current model
                logp_pos_pi, pos_length = compute_masked_logprob(model, tokenizer, initial_sequence, pos_traj, device_t, return_length=True)
                logp_neg_pi, neg_length = compute_masked_logprob(model, tokenizer, initial_sequence, neg_traj, device_t, return_length=True)
                
                # Compute masked log-probs for pos and neg continuations using reference model
                logp_pos_ref, _ = compute_masked_logprob(model_ref, tokenizer, initial_sequence, pos_traj, device_t, return_length=True)
                logp_neg_ref, _ = compute_masked_logprob(model_ref, tokenizer, initial_sequence, neg_traj, device_t, return_length=True)
                
                # For models with device_map="auto", let transformers handle device placement
                # No need to manually move tensors to specific devices
                
                # Average log probabilities to handle length differences
                # Clamp log probabilities to prevent extreme values before averaging
                logp_pos_pi = torch.clamp(logp_pos_pi, min=-50, max=50)
                logp_neg_pi = torch.clamp(logp_neg_pi, min=-50, max=50)
                logp_pos_ref = torch.clamp(logp_pos_ref, min=-50, max=50)
                logp_neg_ref = torch.clamp(logp_neg_ref, min=-50, max=50)
                
                avg_pos_pi = logp_pos_pi / max(1, pos_length)
                avg_neg_pi = logp_neg_pi / max(1, neg_length)
                avg_pos_ref = logp_pos_ref / max(1, pos_length)
                avg_neg_ref = logp_neg_ref / max(1, neg_length)
                
                # DPO loss: stable form using reference model
                diff = ((avg_pos_pi - avg_pos_ref) - (avg_neg_pi - avg_neg_ref)) / tau
                
                # Check for NaN/Inf in the difference before computing loss
                if torch.isnan(diff) or torch.isinf(diff):
                    logger.warning(f"NaN/Inf detected in DPO diff for qid {qid}, skipping this step")
                    continue
                
                # Clamp diff to prevent extreme values in softplus
                diff = torch.clamp(diff, min=-10, max=10)
                loss = F.softplus(-diff)
                # Check for NaN/Inf in the final loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf detected in DPO loss for qid {qid}, skipping this step")
                    continue

                # Gradient accumulation
                loss = loss / gradient_accumulation_steps  # Scale loss by accumulation steps
                loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_count += 1
                
                # Step 3: Update model after accumulating gradients
                if accumulation_count >= gradient_accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accumulated_loss = 0.0
                    accumulation_count = 0
                
                # Update progress bar with current loss and additional info
                current_loss = loss.item() * gradient_accumulation_steps  # Show unscaled loss
                question_pbar.set_postfix({
                    'qid': qid[:20] + '...' if len(qid) > 20 else qid,
                    'loss': f"{current_loss:.4f}",
                    'acc_loss': f"{accumulated_loss:.4f}",
                    'reason': reason[:15] + '...' if len(str(reason)) > 15 else str(reason),
                    'step': global_step,
                    'acc_count': accumulation_count,
                    'pos_len': pos_length,
                    'neg_len': neg_length
                })
                
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step} | qid {qid} | loss {current_loss:.4f} | acc_loss {accumulated_loss:.4f} | reason {reason}")
                
                # Periodic save
                if global_step % save_every_steps == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint_step{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    try:
                        # Save model (works for both full models and LoRA adapters)
                        if lora_only:
                            logger.info(f"Saving LoRA adapter checkpoint to {ckpt_dir}")
                        else:
                            logger.info(f"Saving full model checkpoint to {ckpt_dir}")
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        logger.info(f"Saved checkpoint to {ckpt_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to save checkpoint at step {global_step}: {e}")
                
                # Clear GPU cache periodically to manage memory
                if global_step % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            except Exception as e:
                logger.error(f"Error computing DPO loss for qid {qid}: {e}")
                # Log additional debugging information
                logger.error(f"  - pos_traj length: {len(pos_traj) if pos_traj else 'None'}")
                logger.error(f"  - neg_traj length: {len(neg_traj) if neg_traj else 'None'}")
                logger.error(f"  - reason: {reason}")
                logger.error(f"  - device: {device_t}")
                import traceback
                logger.error(f"  - traceback: {traceback.format_exc()}")
                continue
        
        # Flush any remaining accumulated gradients
        if accumulation_count > 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            logger.info(f"Flushed remaining {accumulation_count} accumulated gradients at end of epoch")
        
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
            "training_mode": "off_policy",
            "training_summary": f"Off-policy DPO training: frozen reasoner generates trajectories, separate training model updated. See dpo_trajectories_used_epoch_{epoch + 1}.json for DPO training pairs."
        }
        
        summary_file = os.path.join(output_dir, f"dpo_training_summary_epoch_{epoch+1}.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(dpo_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved DPO training summary to {summary_file}")

    # Final save
    final_dir = os.path.join(output_dir, "dpo_final")
    os.makedirs(final_dir, exist_ok=True)
    try:
        # Save model (works for both full models and LoRA adapters)
        if lora_only:
            logger.info(f"Saving final LoRA adapter to {final_dir}")
        else:
            logger.info(f"Saving final full model to {final_dir}")
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved final model/adapters to {final_dir}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="OFF-POLICY DPO training for agentic RAG reasoner (uses frozen reasoner for generation, trains separate model)")
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
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--max-data-samples", type=int, default=None, help="Maximum number of data samples to use (None for all)")
    parser.add_argument("--high-randomness", action="store_true", default=True, help="Enable high randomness mode for diverse trajectory generation (default: True)")
    parser.add_argument("--no-high-randomness", dest="high_randomness", action="store_false", help="Disable high randomness mode")
    parser.add_argument("--lora-only", action="store_true", default=False, help="Train LoRA adapter only (not full model)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of gradient accumulation steps before optimizer step")
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
        lora_only=args.lora_only,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
