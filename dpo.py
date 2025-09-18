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

from main import InferenceSystem, InferenceConfig
from utils import load_dataset, is_exact_match, extract_answer, cover_match
from models import get_model_device
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

    # Build full input ids = prompt + continuation and move to the specified device
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1).to(device)
    
    cont_start = prompt_ids.shape[1]
    seq_len = input_ids.shape[1]

    # Forward pass (keep grad)
    outputs = model(input_ids, return_dict=True)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Create position tensors directly on the specified device to avoid device mismatch
    pred_pos = torch.arange(cont_start - 1, seq_len - 1, device=device, dtype=torch.long)
    target_pos = torch.arange(cont_start, seq_len, device=device, dtype=torch.long)

    logits_preds = logits[0, pred_pos, :]  # shape (num_cont_tokens, vocab)
    target_ids = input_ids[0, target_pos].long()  # shape (num_cont_tokens,) - ensure LongTensor

    log_probs = F.log_softmax(logits_preds, dim=-1)
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # (num_cont_tokens,)

    # Build mask: True for tokens to KEEP (not in <information>), False to drop
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
    
    # Sum log-probs over kept tokens
    if keep_mask.sum().item() == 0:
        # no tokens to train on in this continuation
        logger.warning(f"All tokens masked out in continuation. Total tokens: {len(keep_mask)}, Info spans: {len(info_spans)}")
        
        
        if return_length:
            return torch.tensor(0.0, device=token_log_probs.device, dtype=torch.float32), 0
        return torch.tensor(0.0, device=token_log_probs.device, dtype=torch.float32)

    kept_log_probs = token_log_probs[keep_mask]
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

    # Use the SAME reasoner model from InferenceSystem for training (true on-policy training)
    logger.info("Using the same reasoner model from InferenceSystem for training...")
    model = system.reasoner.model  # This is the exact same model used for generation
    
    # MERGE LoRA ADAPTER INTO BASE MODEL (if LoRA is present)
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
    
    # Set model to training mode
    model.train()
    
    # Enable all parameters for training (full fine-tuning after LoRA merge)
    for param in model.parameters():
        param.requires_grad = True
    
    # Prepare optimizer for all parameters (full model training)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable params: {sum(p.numel() for p in trainable):,}")
    logger.info("Training full model (LoRA merged + base model parameters)")
    
    optimizer = AdamW(trainable, lr=lr)
    
    

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
                
                
                # Compute masked log-probs for pos and neg continuations
                logp_pos, pos_length = compute_masked_logprob(model, tokenizer, initial_sequence, pos_traj, device_t, return_length=True)
                logp_neg, neg_length = compute_masked_logprob(model, tokenizer, initial_sequence, neg_traj, device_t, return_length=True)
                
                # Move tensors to device
                logp_pos = logp_pos.to(device_t)
                logp_neg = logp_neg.to(device_t)
                
                # Per-token DPO loss: average log probabilities to handle length differences
                avg_logp_pos = logp_pos / pos_length if pos_length > 0 else logp_pos
                avg_logp_neg = logp_neg / neg_length if neg_length > 0 else logp_neg
                
                # DPO loss: stable form (softplus) on per-token basis
                diff = (avg_logp_pos - avg_logp_neg) / (tau if tau > 0 else 1.0)
                loss = F.softplus(-diff)  # equals -log sigmoid(diff) in a stable way

                
                # Step 3: Backpropagate and update model immediately (on-policy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                global_step += 1
                
                # Update progress bar with current loss and additional info
                question_pbar.set_postfix({
                    'qid': qid[:20] + '...' if len(qid) > 20 else qid,
                    'loss': f"{loss.item():.4f}",
                    'reason': reason[:15] + '...' if len(str(reason)) > 15 else str(reason),
                    'step': global_step,
                    'pos_len': pos_length,
                    'neg_len': neg_length
                })
                
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step} | qid {qid} | loss {loss.item():.4f} | reason {reason}")
                
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
