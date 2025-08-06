import os
# Set JAX to use CUDA GPU backend specifically (avoiding ROCm initialization issues)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['JAX_ENABLE_X64'] = 'True'

import flashrag.retriever  # HACK: although not used, importing this before importing load_dataset somehow make sure get_retriever() can work. Dependency bug of flashrag/faiss/datasets.
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from omegaconf import OmegaConf
import hydra
from hydra.utils import to_absolute_path

from flashrag.config import Config
from flashrag.utils import get_retriever

from rar_evaluate import run_evaluation
from rar_generator import GeneratorMixin
from utils import (
    load_user_message_template,
    load_sys_message, 
    extract_reasoning_and_answer,
    parse_reasoning_generation,
    parse_summary_generation
)

import re
import json
import logging
from tqdm import tqdm
import datetime
import time
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Constants for XML-like tags
BEGIN_OF_THINK, END_OF_THINK = "<think>", "</think>"
BEGIN_OF_ANSWER, END_OF_ANSWER = "<answer>", "</answer>"
BEGIN_OF_SEARCH, END_OF_SEARCH = "<search>", "</search>"
BEGIN_OF_INFO, END_OF_INFO = "<information>", "</information>"

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
        # Ensure we have the target layer
        if self.target_layer >= len(hidden_states):
            logger.warning(f"Target layer {self.target_layer} is out of bounds. Using last layer {len(hidden_states)-1}")
            target_layer_idx = len(hidden_states) - 1
        else:
            target_layer_idx = self.target_layer
        
        # Get the target layer hidden states
        target_layer_hidden = hidden_states[target_layer_idx]
        
        # If target_positions is provided, use those specific positions
        # Otherwise, fall back to the last token (for backward compatibility)
        if target_positions is not None:
            # Extract hidden states at specific target positions
            batch_size = target_layer_hidden.shape[0]
            target_hidden = torch.zeros(batch_size, target_layer_hidden.shape[-1], 
                                      device=target_layer_hidden.device, 
                                      dtype=target_layer_hidden.dtype)
            
            for i, pos in enumerate(target_positions):
                if pos < target_layer_hidden.shape[1]:
                    target_hidden[i] = target_layer_hidden[i, pos, :]
                else:
                    # Fallback to last position if target position is out of bounds
                    target_hidden[i] = target_layer_hidden[i, -1, :]
        else:
            # Take the last token's hidden state from the target layer (original behavior)
            target_hidden = target_layer_hidden[:, -1, :]
        
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
        
        # Convert all hidden states to float32 and ensure they're on the same device as consistency head
        consistency_head_device = next(self.consistency_head.parameters()).device
        hidden_states = []
        for hs in outputs.hidden_states:
            # Move hidden states to the same device as consistency head and convert to float32
            hs_processed = hs.to(device=consistency_head_device, dtype=torch.float32)
            hidden_states.append(hs_processed)
        
        # Predict consistency score with target positions
        consistency_score = self.consistency_head(hidden_states, target_positions)
        
        return {
            "logits": outputs.logits,
            "consistency_score": consistency_score
        }

def load_model_with_consistency_head(
    model_path: str, 
    consistency_head_path: str,
    use_quantization: bool = True, 
    use_multi_gpu: bool = False,
    target_layer: int = 40
):
    """Load model with consistency head."""
    logger.info(f"Loading model from {model_path}...")
    logger.info(f"Use quantization: {use_quantization}")
    logger.info(f"Use multi-GPU: {use_multi_gpu}")
    
    # Set up quantization if requested
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.float16,  # Use FP16 for storage
            llm_int8_threshold=6.0,  # More aggressive int8 threshold
        )
    else:
        quantization_config = None
    
    # Set up device mapping for multi-GPU or single GPU
    if use_multi_gpu and torch.cuda.device_count() > 1:
        # Get available GPU memory and distribute layers across GPUs
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using {num_gpus} GPUs for model distribution")
        
        # Create device map for multi-GPU distribution
        # This will automatically distribute layers across available GPUs
        device_map = "auto"
        
        # Set memory limits for each GPU (leave some memory for consistency head)
        max_memory = {}
        for i in range(num_gpus):
            # Get total GPU memory and reserve some for consistency head
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
            reserved_memory = 3.0  # Reserve 3GB for consistency head and other operations
            available_memory = max(1.0, gpu_memory - reserved_memory)  # Ensure at least 1GB is available
            max_memory[i] = f"{available_memory:.0f}GB"
        
        logger.info(f"Memory allocation per GPU: {max_memory}")
    else:
        # Single GPU or CPU
        device_map = {"": 0}
        max_memory = {0: "75GB"} if torch.cuda.is_available() else None
        logger.info("Using single GPU or CPU")
    
    # Load base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16 if not use_quantization else None,
            trust_remote_code=True,
            max_memory=max_memory if use_multi_gpu else max_memory
        )
        logger.info("Base model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load base model: {str(e)}")
        if use_multi_gpu and not use_quantization:
            logger.info("Trying fallback: single GPU with quantization")
            # Fallback to single GPU with quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_storage=torch.float16,
                    llm_int8_threshold=6.0,
                ),
                device_map={"": 0},
                torch_dtype=torch.float16,
                trust_remote_code=True,
                max_memory={0: "75GB"}
            )
            logger.info("Base model loaded with fallback configuration")
        else:
            raise e
    
    # Get hidden size from model
    hidden_size = base_model.config.hidden_size
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Create consistency head
    layer_sizes = [hidden_size, 512, 256, 1]
    logger.info(f"Consistency head layer sizes: {layer_sizes}")
    logger.info(f"Probing layer {target_layer}")
    
    consistency_head = ConsistencyHead(hidden_size, layer_sizes=layer_sizes, target_layer=target_layer)
    
    # Load trained consistency head weights
    if os.path.exists(consistency_head_path):
        consistency_head.load_state_dict(torch.load(consistency_head_path, map_location='cpu'))
        logger.info(f"Loaded consistency head from {consistency_head_path}")
    else:
        logger.warning(f"Consistency head file not found at {consistency_head_path}")
    
    # Create wrapper model
    model = ModelWithConsistencyHead(base_model, consistency_head)
    
    # Handle device placement for consistency head
    if use_multi_gpu and torch.cuda.device_count() > 1:
        # For multi-GPU, place consistency head on the first GPU
        # This ensures it can access the hidden states from the distributed model
        device = torch.device("cuda:0")
        model.consistency_head = model.consistency_head.to(device)
        logger.info(f"Consistency head moved to {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        logger.info("Model moved to GPU")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    consistency_head_params = sum(p.numel() for p in model.consistency_head.parameters())
    logger.info(f"Consistency head parameters: {consistency_head_params:,}")
    logger.info("Base model parameters frozen, only consistency head will be used for inference")
    
    return model, device

def validate_model_loading(model, device, use_multi_gpu):
    """Validate that the model was loaded correctly and can handle inference."""
    logger.info("Validating model loading...")
    
    try:
        # Test a simple forward pass with dummy data
        dummy_input = torch.randint(0, 1000, (1, 10), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=dummy_input)
        
        logger.info("Model validation successful - forward pass works")
        
        # Check if consistency head is accessible
        if hasattr(model, 'consistency_head'):
            consistency_score = outputs.get('consistency_score')
            if consistency_score is not None:
                logger.info(f"Consistency head working - output shape: {consistency_score.shape}")
            else:
                logger.warning("Consistency head output not found in model outputs")
        
        # Log device information
        if use_multi_gpu and torch.cuda.device_count() > 1:
            logger.info("Multi-GPU setup detected")
            for i in range(torch.cuda.device_count()):
                gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_memory_used:.2f}GB / {gpu_memory_total:.2f}GB used")
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def find_closing_search_position(text: str, tokenizer: AutoTokenizer) -> int:
    """Find the position of the '>' token at the end of </search>."""
    # Look for the pattern </search> in the text
    search_end_pattern = r"</search>"
    matches = list(re.finditer(search_end_pattern, text))
    
    if matches:
        # Get the last occurrence
        last_match = matches[-1]
        end_pos = last_match.end()
        
        # Tokenize the text up to this position
        tokens = tokenizer.encode(text[:end_pos])
        
        # Find the position of the '>' token
        # We need to find the last '>' in the tokenized sequence
        text_tokens = tokenizer.convert_ids_to_tokens(tokens)
        
        # Look for the '>' token in the last few tokens
        for i in range(len(text_tokens) - 1, max(0, len(text_tokens) - 10), -1):
            if text_tokens[i] == '>':
                return i
        
        # If not found, return the last token position
        return len(tokens) - 1
    
    return -1

def format_initial_qa_prompt(
    item: Dict[str, Any], 
    user_message_template: str, 
    sys_message: str, 
    tokenizer: Optional[AutoTokenizer], 
    use_openai_format: bool = False
) -> Dict[str, Any]:
    """
    Format initial QA prompt based on the model type.
    
    Args:
        item: Data item containing the question.
        user_message_template: Template for user message.
        sys_message: System message.
        tokenizer: Tokenizer for local models.
        use_openai_format: Whether to use OpenAI format.
        
    Returns:
        Formatted prompt or messages dict.
    """
    user_message = user_message_template.format(question=item["question"])
    
    if use_openai_format:
        messages = []
        if sys_message:
            messages.append({"role": "system", "content": sys_message})
        messages.append({"role": "user", "content": user_message})
        return {"messages": messages}
    else:
        messages = []
        if sys_message:
            messages.append({"role": "system", "content": sys_message})
        messages.append({"role": "user", "content": user_message})
        
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {"prompt": prompt}
        else:
            return {"messages": messages}

def format_retrieval_summary_prompts(
    search_queries: List[str], 
    search_results: List[List[Dict[str, Any]]], 
    user_message_template: str, 
    sys_message: Optional[str], 
    tokenizer: Optional[AutoTokenizer], 
    use_openai_format: bool = False
) -> List[Union[str, List[Dict[str, str]]]]:
    """
    Format retrieval summary prompts.
    
    Args:
        search_queries: List of search queries.
        search_results: List of search result lists.
        user_message_template: Template for user message.
        sys_message: System message.
        tokenizer: Tokenizer for local models.
        use_openai_format: Whether to use OpenAI format.
        
    Returns:
        List of formatted prompts or message lists.
    """
    def format_doc(doc: Dict[str, Any]) -> str:
        title = doc.get('title', doc.get('id', 'Unknown'))
        text = doc.get('text', doc.get('content', str(doc)))
        return f"Document (Title: {title}): {text}"
    
    prompts = []
    for query, results in zip(search_queries, search_results):
        retrieved_context = "\n\n".join([format_doc(result) for result in results])
        user_message = user_message_template.format(question=query, documents=retrieved_context)
        sys_content = sys_message if sys_message is not None else "You are a helpful assistant."
        
        if use_openai_format:
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_message},
            ]
            prompts.append(messages)
        else:
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_message},
            ]
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
            else:
                prompts.append(messages)
    
    return prompts

def build_reasoner_prompt(
    item: Dict[str, Any], 
    sys_message: str, 
    user_message_template: str,
    cfg: Any,
    tokenizer: Optional[AutoTokenizer]
) -> Union[List[Dict[str, str]], Dict[str, str]]:
    """Build reasoner prompt based on model mode."""
    if cfg.model.reasoner_mode in ["openai", "proxy"]:
        messages = []
        if sys_message:
            messages.append({"role": "system", "content": sys_message})
        messages.append({"role": "user", "content": user_message_template.format(question=item["question"])})
        return messages
    else:
        messages = []
        if sys_message:
            messages.append({"role": "system", "content": sys_message})
        messages.append({"role": "user", "content": user_message_template.format(question=item["question"])})
        
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {"prompt": prompt}
        else:
            return {"messages": messages}

def build_summarizer_prompt(
    search_queries: List[str], 
    search_results: List[List[Dict[str, Any]]], 
    user_message_template: str, 
    sys_message: Optional[str],
    cfg: Any,
    tokenizer: Optional[AutoTokenizer]
) -> Union[List[Dict[str, str]], Dict[str, str]]:
    """Build summarizer prompt based on model mode."""
    if cfg.model.summarizer_mode in ["openai", "proxy"]:
        messages = []
        if sys_message:
            messages.append({"role": "system", "content": sys_message})
        user_content = user_message_template.format(question=search_queries[0], documents="")
        messages.append({"role": "user", "content": user_content})
        return messages
    else:
        def format_doc(doc: Dict[str, Any]) -> str:
            title = doc.get('title', doc.get('id', 'Unknown'))
            text = doc.get('text', doc.get('content', str(doc)))
            return f"Document (Title: {title}): {text}"
        
        retrieved_context = "\n\n".join([format_doc(result) for result in search_results[0]])
        user_content = user_message_template.format(question=search_queries[0], documents=retrieved_context)
        messages = [{"role": "system", "content": user_content}]
        
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {"prompt": prompt}
        else:
            return {"messages": messages}

def process_per_document_summarization(
    search_queries: List[str],
    search_results: List[List[Dict[str, Any]]],
    summarizer_user_message_template: str,
    summarizer_model: GeneratorMixin,
    summarizer_sampling_params: Dict[str, Any],
    tokenizer: Optional[AutoTokenizer],
    use_openai_summarizer: bool,
    turn: int,
    consistency_scores: List[float] = None
) -> List[Dict[str, Any]]:
    """Process per-document summarization and aggregation."""
    retrieval_history_entries = []
    
    for query, results in zip(search_queries, search_results):
        per_doc_prompts = []
        doc_titles = []
        
        for doc in results:
            # Extract document title and text
            if 'title' in doc and 'text' in doc:
                doc_title = doc['title']
                doc_text = doc['text']
            elif 'title' in doc and 'contents' in doc:
                doc_title = doc['title']
                doc_text = doc['contents']
            elif 'text' in doc:
                doc_title = ''
                doc_text = doc['text']
            elif 'contents' in doc:
                doc_title = ''
                doc_text = doc['contents']
            else:
                doc_title = ''
                doc_text = str(doc)
            
            # Filter for extracted information sections
            extracted_sections = []
            for line in doc_text.splitlines():
                if line.strip().startswith('### Extracted Information'):
                    extracted_sections.append(line)
            
            filtered_text = '\n'.join(extracted_sections) if extracted_sections else ''
            doc_titles.append(doc_title)
            
            user_message = summarizer_user_message_template.format(
                question=query, 
                documents=f"Document (Title: {doc_title}): {filtered_text}"
            )
            messages = [{"role": "system", "content": user_message}]
            
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = messages
            per_doc_prompts.append(prompt)
        
        # Generate per-document summaries
        per_doc_outputs_raw = []
        batch_size = 3
        for i in range(0, len(per_doc_prompts), batch_size):
            batch_prompts = per_doc_prompts[i:i+batch_size]
            if not isinstance(batch_prompts, list):
                batch_prompts = [batch_prompts]
            per_doc_outputs_raw.extend(
                summarizer_model.generate(
                    batch_prompts, 
                    {**summarizer_sampling_params, 'batch_size': batch_size}
                )
            )
        
        per_doc_summaries = [parse_summary_generation(out["text"])[1] for out in per_doc_outputs_raw]
        
        # Aggregate summaries
        summaries_context = "\n\n".join([
            f"Summary for Document {i+1} (Title: {doc_titles[i]}): {per_doc_summaries[i]}" 
            for i in range(len(per_doc_summaries))
        ])
        
        agg_user_message = (
            "Given the following summaries of retrieved documents, write the briefest possible answer to the user query, "
            "containing only the most essential information. Do not include any unnecessary details.\n"
            "You must only use information from the summaries.\n"
            "\n"
            "### User Query\n"
            f"{query}\n"
            "\n"
            "### Document Summaries\n"
            f"{summaries_context}"
        )
        
        agg_messages = [{"role": "system", "content": agg_user_message}]
        
        if tokenizer is not None:
            agg_prompt = tokenizer.apply_chat_template(
                agg_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            agg_prompt = agg_messages
        
        # Generate final summary
        final_summary_output_raw = summarizer_model.generate([agg_prompt], summarizer_sampling_params)[0]
        final_summary_output_raw_tagged = f"### Extracted Information\n{final_summary_output_raw['text'].strip()}"
        final_analysis, final_summary = parse_summary_generation(final_summary_output_raw_tagged)
        
        # Get consistency score for this query if available
        query_index = search_queries.index(query)
        consistency_score = consistency_scores[query_index] if consistency_scores and query_index < len(consistency_scores) else None
        
        retrieval_history_entries.append({
            "query": query,
            "per_doc_summaries": per_doc_summaries,
            "per_doc_raw_outputs": [out["text"] for out in per_doc_outputs_raw],
            "final_summary": final_summary,
            "final_summary_raw_output": final_summary_output_raw['text'],
            "prompt_and_output": (
                agg_prompt if tokenizer is None 
                else agg_prompt + "\n\n[OUTPUT STARTS HERE]\n\n" + final_analysis + "\n\n" + final_summary
            ),
            "summarization_method": "fallback",
            "retrieval_method": "normal_retrieval",
            "consistency_score": consistency_score,
            "turn": turn
        })
    
    return retrieval_history_entries

def create_simple_retrieval_history_entries(
    search_queries: List[str],
    search_results: List[List[Dict[str, Any]]],
    summary_outputs: List[Tuple[str, str]],
    summarizer_prompt_list: List[Union[str, List[Dict[str, str]]]],
    summarizer_output: List[Dict[str, Any]],
    turn: int,
    consistency_scores: List[float] = None
) -> List[Dict[str, Any]]:
    """Create retrieval history entries for simple summarization case."""
    retrieval_history_entries = []
    
    for i, (query, results, summary_output, prompt, raw_output) in enumerate(zip(
        search_queries, search_results, summary_outputs, summarizer_prompt_list, summarizer_output
    )):
        # Format documents for history
        formatted_docs = []
        for doc in results:
            if 'title' in doc and 'text' in doc:
                doc_title = doc['title']
                doc_text = doc['text']
            elif 'title' in doc and 'contents' in doc:
                doc_title = doc['title']
                doc_text = doc['contents']
            elif 'text' in doc:
                doc_title = ''
                doc_text = doc['text']
            elif 'contents' in doc:
                doc_title = ''
                doc_text = doc['contents']
            else:
                doc_title = ''
                doc_text = str(doc)
            formatted_docs.append(f"Document (Title: {doc_title}): {doc_text}")
        
        # Get consistency score for this query if available
        consistency_score = consistency_scores[i] if consistency_scores and i < len(consistency_scores) else None
        
        retrieval_history_entries.append({
            "query": query,
            "retrieved_documents": formatted_docs,
            "final_summary": summary_output[1].strip(),
            "final_summary_raw_output": raw_output["text"],
            "prompt_and_output": (
                str(prompt) + "\n\n[OUTPUT STARTS HERE]\n\n" + summary_output[0] + "\n\n" + summary_output[1]
            ),
            "summarization_method": "simple",
            "retrieval_method": "normal_retrieval",
            "consistency_score": consistency_score,
            "turn": turn
        })
    
    return retrieval_history_entries

def create_model_knowledge_fallback_entries(
    sequences_to_skip_retrieval: List[int],
    active_sequences: List[Dict[str, Any]],
    reasoner_outputs: List[Dict[str, Any]],
    consistency_scores: List[float],
    turn: int
) -> List[Dict[str, Any]]:
    """Create retrieval history entries for model knowledge fallback case."""
    retrieval_history_entries = []
    
    for i in sequences_to_skip_retrieval:
        if i < len(active_sequences) and i < len(reasoner_outputs):
            seq = active_sequences[i]
            output = reasoner_outputs[i]
            consistency_score = consistency_scores[i] if i < len(consistency_scores) else 0.0
            
            # Get the search query that was stored when the sequence was marked for skipping
            search_q = seq.get("model_knowledge_search_query", "No search query stored")
            
            # Extract the reasoning from the output
            generated = output["text"]
            thought, _, answer = parse_reasoning_generation(generated)
            
            retrieval_history_entries.append({
                "query": search_q,
                "retrieved_documents": [],  # No documents retrieved
                "final_summary": "Used model's direct knowledge",
                "final_summary_raw_output": generated,
                "prompt_and_output": f"Model knowledge fallback - Consistency score: {consistency_score:.4f}\n\n{generated}",
                "summarization_method": "model_knowledge_fallback",
                "consistency_score": consistency_score,
                "retrieval_method": "model_knowledge",
                "turn": turn
            })
    
    return retrieval_history_entries

def save_outputs(
    data: List[Dict[str, Any]], 
    all_sequences: List[Dict[str, Any]], 
    metrics_list: List[Dict[str, float]], 
    retrieval_history: List[Dict[str, Any]], 
    metrics_agg: Dict[str, float], 
    output_dir: str
) -> None:
    """Save all outputs to files."""
    t = datetime.datetime.now()
    timestamp = t.strftime("%m-%d-%H-%M-%S")
    
    # Save main outputs
    outputs = []
    for item, seq, metrics in zip(data, all_sequences, metrics_list):
        outputs.append({
            "prompt": seq.get("prompt"),
            "messages": seq.get("messages"),
            "output": seq["output"],
            "question": item["question"],
            "pred": seq["answer"],
            "gold": item["golden_answers"],
            "finished": seq["finished"],
            "metrics": metrics,
        })
    
    output_file_name = f'output_{timestamp}.json'
    with open(f"{output_dir}/{output_file_name}", "w") as f:
        json.dump(outputs, f, indent=2)
    
    # Save retrieval history
    logger.info(f"Saving {len(retrieval_history)} retrieval history entries")
    retrieval_history_file_name = f'retrieval_history_{timestamp}.json'
    with open(f"{output_dir}/{retrieval_history_file_name}", "w") as f:
        json.dump(retrieval_history, f, indent=2)
    
    # Save metrics
    metrics_file_name = f'metrics_{timestamp}.json'
    with open(f"{output_dir}/{metrics_file_name}", "w") as f:
        json.dump(metrics_agg, f, indent=2)

def create_metrics_plot(
    metrics_list: List[Dict[str, float]], 
    output_dir: str
) -> None:
    """Create and save metrics plot."""
    retrieval_to_metrics = defaultdict(list)
    for m in metrics_list:
        retrieval_to_metrics[m['num_retrieval']].append(m)
    
    retrieval_counts = sorted(retrieval_to_metrics.keys())
    avg_em = []
    avg_cover = []
    avg_f1 = []
    
    for rc in retrieval_counts:
        ms = retrieval_to_metrics[rc]
        avg_em.append(sum(x['em'] for x in ms) / len(ms))
        avg_cover.append(sum(x['cover_match'] for x in ms) / len(ms))
        avg_f1.append(sum(x['str_f1'] for x in ms) / len(ms))
    
    plt.figure(figsize=(8, 6))
    plt.plot(retrieval_counts, avg_em, marker='o', label='EM')
    plt.plot(retrieval_counts, avg_cover, marker='s', label='Cover Match')
    plt.plot(retrieval_counts, avg_f1, marker='^', label='String F1')
    plt.xlabel('Number of Retrievals')
    plt.ylabel('Average Metric')
    plt.title('Average Metrics by Retrieval Count')
    plt.legend()
    plt.grid(True)
    
    plot_path = f"{output_dir}/metrics_by_retrieval_count.png"
    plt.savefig(plot_path)
    logger.info(f"Saved retrieval-metrics plot to {plot_path}")

@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: Any) -> None:
    """
    Main inference function for RAR (Reasoning and Retrieval) pipeline.
    
    Args:
        cfg: Hydra configuration object containing all experiment parameters.
    """
    logger.info("***Experiment Config***")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("***********************")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.1f}GB")
    else:
        logger.info("CUDA not available, using CPU")

    # Load dataset
    dataset_path = os.path.join(cfg.data.dataset_root_dir, cfg.data.dataset_name, f"{cfg.data.dataset_split}.jsonl")
    data = load_dataset("json", data_files=dataset_path, split="train")

    # Apply debug and subset limits
    debug_num_examples = cfg.data.get('debug_num_examples')
    if debug_num_examples is not None:
        data = data.select(range(min(debug_num_examples, len(data))))

    if cfg.data.subset_size:
        data = data.shuffle(seed=42).select(range(cfg.data.subset_size))
        logger.info(f"Using a subset of the data: {cfg.data.subset_size}")
    
    # Initialize tokenizers and models
    use_openai_reasoner = cfg.model.reasoner_mode == "openai"
    use_openai_summarizer = cfg.model.summarizer_mode == "openai"
    
    # Load single tokenizer for the model
    tokenizer = None if use_openai_reasoner else AutoTokenizer.from_pretrained(cfg.model.reasoner_name_or_path)
    
    # Load single model with consistency head if using local model
    consistency_model = None
    device = None
    if not use_openai_reasoner:
        consistency_head_path = cfg.model.get('consistency_head_path', './trained_consistency_head_40.pt')
        target_layer = cfg.model.get('target_layer', 40)
        # Get model loading settings from config
        use_quantization = cfg.model.get('use_quantization', False)
        use_multi_gpu = cfg.model.get('use_multi_gpu', True)
        
        logger.info("Loading single Qwen3-32B model with consistency head...")
        logger.info("This model will be reused for reasoner and summarizer tasks - no additional model loading!")
        consistency_model, device = load_model_with_consistency_head(
            cfg.model.reasoner_name_or_path,
            consistency_head_path,
            use_quantization=use_quantization,
            use_multi_gpu=use_multi_gpu,
            target_layer=target_layer
        )
        
        # Validate the loaded model
        if not validate_model_loading(consistency_model, device, use_multi_gpu):
            logger.error("Model validation failed. Please check your configuration and GPU setup.")
            raise RuntimeError("Model validation failed")
        
        logger.info("Single model loaded successfully - will be reused for reasoner and summarizer")
    
    # Create reasoner and summarizer models that reuse the same loaded model
    if not use_openai_reasoner and consistency_model is not None:
        # Create custom generators that reuse the loaded model
        from rar_generator import GeneratorMixin
        
        class ReuseModelGeneratorMixin(GeneratorMixin):
            """GeneratorMixin that reuses an already loaded model."""
            
            def __init__(self, mode: str, loaded_model, loaded_tokenizer, device, **kwargs):
                self.mode = mode
                self.model = loaded_model.base_model  # Use the base model from consistency model
                self.tokenizer = loaded_tokenizer
                self.device = device
                self.server_url = kwargs.get('server_url')
                self.server_params = kwargs.get('server_params', {})
                
                # Skip the normal initialization that would load a new model
                logger.info(f"Reusing loaded model for {mode} generator")
            
            def _init_offline_model(self):
                """Skip model loading since we're reusing an existing model."""
                pass
            
            def _init_transformers_pipeline(self):
                """Skip pipeline initialization since we're reusing an existing model."""
                pass
            
            def generate(self, prompts, sampling_params=None):
                """Generate using the reused model."""
                if sampling_params is None:
                    sampling_params = {}
                
                # Convert prompts to the format expected by the model
                if isinstance(prompts[0], dict) and 'messages' in prompts[0]:
                    # OpenAI format - convert to prompt
                    processed_prompts = []
                    for prompt_dict in prompts:
                        if 'messages' in prompt_dict:
                            messages = prompt_dict['messages']
                            prompt = self.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            processed_prompts.append(prompt)
                        else:
                            processed_prompts.append(prompt_dict)
                else:
                    processed_prompts = prompts
                
                # Handle stop tokens
                stop_tokens = sampling_params.get('stop', [])
                
                # Generate using the model
                outputs = []
                for i, prompt in enumerate(processed_prompts):
                    logger.info(f"Generating for prompt {i}, max_new_tokens: {sampling_params.get('max_new_tokens', 512)}")
                    logger.info(f"no_stop_trim: {sampling_params.get('no_stop_trim', False)}")
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        # Get all sampling parameters
                        gen_kwargs = {
                            **inputs,
                            'max_new_tokens': sampling_params.get('max_new_tokens', 512),
                            'temperature': sampling_params.get('temperature', 0.6),
                            'top_p': sampling_params.get('top_p', 0.95),
                            'repetition_penalty': sampling_params.get('repetition_penalty', 1.0),
                            'do_sample': True,
                            'pad_token_id': self.tokenizer.eos_token_id,
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'early_stopping': False,
                        }
                        
                        # Handle no_stop_trim parameter
                        if sampling_params.get('no_stop_trim', False):
                            # If no_stop_trim is True, we don't want to stop at any tokens
                            # Remove eos_token_id to prevent early stopping
                            gen_kwargs.pop('eos_token_id', None)
                        
                        generated_ids = self.model.generate(**gen_kwargs)
                    
                    # Decode the generated text
                    generated_text = self.tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    logger.info(f"Generated text for prompt {i}: {repr(generated_text)}")
                    
                    # Apply stop token filtering manually if needed
                    # Only apply stop token filtering if no_stop_trim is False
                    if stop_tokens and not sampling_params.get('no_stop_trim', False):
                        logger.info(f"Applying stop token filtering for prompt {i}")
                        for stop_token in stop_tokens:
                            if stop_token in generated_text:
                                # Find the first occurrence of the stop token
                                stop_index = generated_text.find(stop_token)
                                if stop_index != -1:
                                    logger.info(f"Found stop token '{stop_token}' at position {stop_index}")
                                    generated_text = generated_text[:stop_index]
                                    break
                    else:
                        logger.info(f"No stop token filtering applied for prompt {i} (no_stop_trim: {sampling_params.get('no_stop_trim', False)})")
                    
                    logger.info(f"Final generated text for prompt {i}: {repr(generated_text)}")
                    outputs.append({"text": generated_text})
                
                return outputs
            
            def shutdown(self):
                """No need to shutdown since we don't own the model."""
                pass
        
        # Create generators that reuse the loaded model
        reasoner_model = ReuseModelGeneratorMixin(
            mode=cfg.model.reasoner_mode,
            loaded_model=consistency_model,
            loaded_tokenizer=tokenizer,
            device=device,
            server_url=cfg.model.reasoner_url,
            server_params=OmegaConf.to_container(cfg.inference.server_params),
        )
        
        summarizer_model = ReuseModelGeneratorMixin(
            mode=cfg.model.summarizer_mode,
            loaded_model=consistency_model,
            loaded_tokenizer=tokenizer,
            device=device,
            server_url=cfg.model.summarizer_url,
            server_params=OmegaConf.to_container(cfg.inference.server_params),
        )
        
        logger.info("Created reasoner and summarizer models that reuse the loaded consistency model")
    else:
        # Fallback to original GeneratorMixin for OpenAI mode or when no consistency model
        reasoner_model = GeneratorMixin(
            mode=cfg.model.reasoner_mode,
            server_url=cfg.model.reasoner_url,
            model_name_or_path=cfg.model.reasoner_name_or_path,
            server_params=OmegaConf.to_container(cfg.inference.server_params),
        )

        summarizer_model = GeneratorMixin(
            mode=cfg.model.summarizer_mode,
            server_url=cfg.model.summarizer_url,
            model_name_or_path=cfg.model.summarizer_name_or_path,
            server_params=OmegaConf.to_container(cfg.inference.server_params),
        )

    # Load templates and system messages
    reasoner_sys_message = load_sys_message(cfg.model.reasoner_name_or_path)
    reasoner_user_message_template = load_user_message_template(
        cfg.inference.prompt_templates_dir, 
        cfg.inference.user_message_template_name
    )
    summarizer_user_message_template = load_user_message_template(
        cfg.inference.prompt_templates_dir, 
        "default_retrieval_summary"
    )
    
    # Format initial prompts
    data = data.map(
        lambda x: format_initial_qa_prompt(
            x, reasoner_user_message_template, reasoner_sys_message, tokenizer, use_openai_reasoner
        ), 
        batched=False
    )

    # Initialize sequences
    if use_openai_reasoner:
        all_sequences = [
            {
                "messages": item["messages"].copy(),
                "finished": False,
                "output": "",
                "answer": "",
            } for item in data
        ]
    else:
        all_sequences = [
            {
                "prompt": item["prompt"],
                "finished": False,
                "output": "",
                "answer": "",
            } for item in data
        ]
   
    logger.info("Example Prompt or Messages:")
    if "prompt" in all_sequences[0]:
        logger.info(all_sequences[0]["prompt"])
    elif "messages" in all_sequences[0]:
        logger.info(all_sequences[0]["messages"])
    else:
        logger.info("No prompt or messages found in first sequence.")

    # Get output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Initialize sampling parameters
    server_params = OmegaConf.to_container(cfg.inference.server_params)
    reasoner_sampling_params = OmegaConf.to_container(cfg.inference.reasoner_sampling_params)
    if tokenizer is not None:
        reasoner_sampling_params["stop"] = [END_OF_SEARCH, END_OF_ANSWER, tokenizer.eos_token]
    else:
        reasoner_sampling_params.pop("stop", None)
    summarizer_sampling_params = OmegaConf.to_container(cfg.inference.summarizer_sampling_params)
    
    # Initialize retriever
    retriever_config = Config(config_dict=OmegaConf.to_container(cfg.retrieval.flashrag_config))
    retriever = get_retriever(retriever_config)

    retrieval_history = []
    
    # Load corpus for index lookup
    corpus_path = cfg.retrieval.flashrag_config.corpus_path
    with open(corpus_path, "r") as f:
        full_corpus = [json.loads(line) for line in f]
    
    # Ensure compatibility with evaluation
    for item in data:
        if 'gold_answer' not in item and 'golden_answers' in item:
            item['gold_answer'] = item['golden_answers']

    # Consistency threshold
    consistency_threshold = cfg.model.get('consistency_threshold', 0.7)
    logger.info(f"Using consistency threshold: {consistency_threshold}")

    # Main inference loop
    turn = 0
    active_sequences = all_sequences
    
    logger.info(f"Example Summarizer {'Messages' if use_openai_summarizer else 'Prompt'}: {summarizer_user_message_template}")
    
    while turn < cfg.inference.max_turns:
        if not active_sequences:
            break
        turn += 1
        logger.info(f"-------------- Turn {turn} --------------")
        logger.info(f"# Active Sequences: {len(active_sequences)}")

        # Reasoner generation
        if use_openai_reasoner:
            reasoner_prompts = [seq["messages"] for seq in active_sequences]
        else:
            reasoner_prompts = [seq["prompt"] for seq in active_sequences]
        
        start_time = time.time()
        reasoner_outputs = reasoner_model.generate(reasoner_prompts, sampling_params=reasoner_sampling_params)
        gen_time = time.time() - start_time
        logger.info(f"Reasoning Generation time: {gen_time:.2f} seconds")

        # Process reasoner outputs and check consistency
        search_queries = []
        sequences_to_skip_retrieval = []
        consistency_scores = [] # Track consistency scores for fallback
        
        for i, (seq, out) in enumerate(zip(active_sequences, reasoner_outputs)):
            generated = out["text"]
            seq["output"] += generated
            thought, search_q, answer = parse_reasoning_generation(generated)
            
            if search_q:
                logger.info(f"Sequence {i} generated search query: {search_q}")
                # Check if we should use consistency head to decide on retrieval
                if not use_openai_reasoner and consistency_model is not None:
                    try:
                        # Find the position of the '>' token at the end of </search>
                        closing_pos = find_closing_search_position(generated, tokenizer)
                        
                        if closing_pos >= 0:
                            # Get the full text up to the closing position
                            full_text = seq["prompt"] + generated
                            tokens = tokenizer.encode(full_text)
                            
                            if closing_pos < len(tokens):
                                # Create input for consistency head
                                input_ids = torch.tensor([tokens], device=device)
                                
                                with torch.no_grad():
                                    outputs = consistency_model(
                                        input_ids=input_ids,
                                        target_positions=[closing_pos]
                                    )
                                    consistency_score = outputs["consistency_score"].item()
                                
                                logger.info(f"Consistency score for sequence {i}: {consistency_score:.4f}")
                                
                                # If consistency score is high, skip retrieval and let reasoner answer directly
                                if consistency_score > consistency_threshold:
                                    logger.info(f"High consistency score ({consistency_score:.4f} > {consistency_threshold}), skipping retrieval for sequence {i}")
                                    sequences_to_skip_retrieval.append(i)
                                    consistency_scores.append(consistency_score) # Store for fallback
                                    
                                    # Store the search query for model knowledge fallback recording
                                    seq["model_knowledge_search_query"] = search_q
                                    
                                    # Continue generation to get answer
                                    continue
                    except Exception as e:
                        logger.warning(f"Error in consistency head inference for sequence {i}: {str(e)}")
                        logger.warning("Continuing with normal retrieval flow")
                        # Continue with normal retrieval if consistency head fails
                
                search_queries.append(search_q)
                if use_openai_reasoner:
                    seq["messages"].append({"role": "assistant", "content": generated.rstrip()})
                else:
                    seq["prompt"] += generated.rstrip()
            elif answer:
                logger.info(f"Sequence {i} generated answer: {answer}")
                seq["finished"] = True
                seq["answer"] = answer
            else:
                logger.info(f"Sequence {i} generated no search query or answer")
                seq["finished"] = True
                seq["answer"] = ""
        
        logger.info(f"Total search queries collected: {len(search_queries)}")
        logger.info(f"Sequences to skip retrieval: {sequences_to_skip_retrieval}")
        
        # Collect all search queries including those that will use model knowledge
        all_search_queries = []
        all_search_queries.extend(search_queries)
        
        # Add search queries from sequences that will use model knowledge
        for i in sequences_to_skip_retrieval:
            if i < len(active_sequences):
                seq = active_sequences[i]
                if "model_knowledge_search_query" in seq:
                    all_search_queries.append(seq["model_knowledge_search_query"])
        
        logger.info(f"Total search queries (including model knowledge): {len(all_search_queries)}")
        
        # Filter out sequences that should skip retrieval
        filtered_active_sequences = []
        filtered_search_queries = []
        
        for i, (seq, search_q) in enumerate(zip(active_sequences, search_queries)):
            if i not in sequences_to_skip_retrieval:
                filtered_active_sequences.append(seq)
                filtered_search_queries.append(search_q)
        
        logger.info(f"Filtered search queries: {len(filtered_search_queries)}")
        
        active_sequences = [seq for seq in active_sequences if not seq["finished"]]
        search_queries = filtered_search_queries

        # Retrieval and summarization
        if search_queries:
            logger.info(f"Processing {len(search_queries)} search queries for retrieval")
            start_search_time = time.time()
            search_results = retriever.batch_search(search_queries)
            ret_time = time.time() - start_search_time
            logger.info(f"Retrieval time: {ret_time:.2f} seconds")
            logger.info(f"Retrieved {len(search_results)} result sets")
            
            # Map search results back to original sequence indices
            # This ensures that search_results[i] corresponds to the correct sequence
            original_search_results = []
            result_idx = 0
            for i in range(len(active_sequences)):
                if i not in sequences_to_skip_retrieval:
                    if result_idx < len(search_results):
                        original_search_results.append(search_results[result_idx])
                        result_idx += 1
                    else:
                        original_search_results.append([])
                else:
                    original_search_results.append([])  # Empty results for skipped sequences
            
            search_results = original_search_results

            # Try simple summarization first
            summarizer_prompt = build_summarizer_prompt(
                search_queries, search_results, summarizer_user_message_template, 
                None, cfg, tokenizer
            )
            # --- FIX: ensure summarizer_prompt_list is always correct for OpenAI/local modes ---
            use_openai_summarizer = cfg.model.summarizer_mode in ["openai", "proxy"]
            if use_openai_summarizer:
                # summarizer_prompt is a list of message dicts or a single message list
                if isinstance(summarizer_prompt, list) and all(isinstance(x, dict) and 'role' in x and 'content' in x for x in summarizer_prompt):
                    summarizer_prompt_list = [summarizer_prompt]
                elif isinstance(summarizer_prompt, list) and all(isinstance(x, list) for x in summarizer_prompt):
                    summarizer_prompt_list = summarizer_prompt
                else:
                    summarizer_prompt_list = [summarizer_prompt]
            else:
                def extract_prompt_str(prompt):
                    if isinstance(prompt, dict):
                        if 'prompt' in prompt:
                            return prompt['prompt']
                        elif 'messages' in prompt:
                            # Join all message contents as a single string
                            return "\n".join([extract_prompt_str(m) for m in prompt["messages"]])
                        elif set(prompt.keys()) == {'role', 'content'}:
                            return prompt['content']
                        else:
                            raise ValueError(f"Unexpected dict keys in prompt: {prompt.keys()}")
                    elif isinstance(prompt, str):
                        return prompt
                    elif isinstance(prompt, list):
                        # If it's a list of dicts or strings, recursively extract
                        return [extract_prompt_str(p) for p in prompt]
                    else:
                        raise ValueError(f"Unexpected prompt type: {type(prompt)}")
                summarizer_prompt_strs = extract_prompt_str(summarizer_prompt)
                # Flatten if nested list
                if isinstance(summarizer_prompt_strs, list) and any(isinstance(x, list) for x in summarizer_prompt_strs):
                    summarizer_prompt_list = [item for sublist in summarizer_prompt_strs for item in (sublist if isinstance(sublist, list) else [sublist])]
                elif isinstance(summarizer_prompt_strs, list):
                    summarizer_prompt_list = summarizer_prompt_strs
                else:
                    summarizer_prompt_list = [summarizer_prompt_strs]
            summarizer_output = summarizer_model.generate(summarizer_prompt_list, summarizer_sampling_params)
            summary_outputs = [parse_summary_generation(out["text"]) for out in summarizer_output]
            
            if summary_outputs:
                # Simple summarization worked
                logger.info(f"Creating simple retrieval history entries for {len(search_queries)} queries")
                retrieval_history_entries = create_simple_retrieval_history_entries(
                    search_queries, search_results, summary_outputs, summarizer_prompt_list, summarizer_output, turn, consistency_scores
                )
                logger.info(f"Created {len(retrieval_history_entries)} retrieval history entries")
                retrieval_history.extend(retrieval_history_entries)
                
                for seq, entry in zip(filtered_active_sequences, retrieval_history_entries):
                    summary = entry["final_summary"].strip()
                    append_summary = f"{BEGIN_OF_INFO} {summary} {END_OF_INFO}\n\n"
                    if use_openai_reasoner:
                        seq["messages"].append({"role": "user", "content": append_summary})
                    else:
                        seq["prompt"] += append_summary
                    seq["output"] += append_summary
            else:
                # Fall back to per-document summarization
                logger.info(f"Creating per-document retrieval history entries for {len(search_queries)} queries")
                retrieval_history_entries = process_per_document_summarization(
                    search_queries, search_results, summarizer_user_message_template,
                    summarizer_model, summarizer_sampling_params, tokenizer, use_openai_summarizer, turn, consistency_scores
                )
                logger.info(f"Created {len(retrieval_history_entries)} per-document retrieval history entries")
                retrieval_history.extend(retrieval_history_entries)
                
                for seq, entry in zip(filtered_active_sequences, retrieval_history_entries):
                    summary = entry["final_summary"].strip()
                    append_summary = f"{BEGIN_OF_INFO} {summary} {END_OF_INFO}\n\n"
                    if use_openai_reasoner:
                        seq["messages"].append({"role": "user", "content": append_summary})
                    else:
                        seq["prompt"] += append_summary
                    seq["output"] += append_summary

        # Create model knowledge fallback entries for this turn
        # These entries record search queries that were solved using the model's knowledge instead of retrieval
        if sequences_to_skip_retrieval:
            logger.info(f"Creating model knowledge fallback entries for {len(sequences_to_skip_retrieval)} sequences")
            # Get the reasoner outputs for sequences that skipped retrieval
            skip_reasoner_outputs = []
            for i in sequences_to_skip_retrieval:
                if i < len(reasoner_outputs):
                    skip_reasoner_outputs.append(reasoner_outputs[i])
                else:
                    skip_reasoner_outputs.append({"text": "No output available"})
            
            fallback_entries = create_model_knowledge_fallback_entries(
                sequences_to_skip_retrieval, active_sequences, skip_reasoner_outputs, consistency_scores, turn
            )
            logger.info(f"Created {len(fallback_entries)} model knowledge fallback entries")
            retrieval_history.extend(fallback_entries)
            
            # Clear the skip lists for next turn
            sequences_to_skip_retrieval.clear()
            consistency_scores.clear()

    # Force finish remaining sequences
    logger.info("-------- Turn limit reached. Forcing active sequences to finish. --------")
    for seq in active_sequences:
        timeout_message = (
            f"Time is up.\n\nSince I have searched for {cfg.inference.max_turns} times, "
            "I am not allowed to search anymore. I should give a final answer now with the information I have."
        )
        if use_openai_reasoner:
            seq["messages"].append({"role": "user", "content": timeout_message})
        else:
            seq["prompt"] += timeout_message
        seq["output"] += timeout_message
    
    if active_sequences:
        if use_openai_reasoner:
            reasoner_prompts = [seq["messages"] for seq in active_sequences]
        else:
            reasoner_prompts = [seq["prompt"] for seq in active_sequences]
        
        reasoner_outputs = reasoner_model.generate(reasoner_prompts, sampling_params=reasoner_sampling_params)
        
        for seq, out in zip(active_sequences, reasoner_outputs):
            generated = out["text"]
            seq["output"] += generated
            thought, search_q, answer = parse_reasoning_generation(generated)
            if answer:
                seq["finished"] = True
                seq["answer"] = answer
            else:
                seq["finished"] = True
                seq["answer"] = ""

    # Model knowledge fallback entries are already created during the loop
    # No need to create them again here

    # Shutdown models
    if reasoner_model is not None:
        reasoner_model.shutdown()
    if summarizer_model is not None:
        summarizer_model.shutdown()

    # Evaluate results
    metrics_list, metrics_agg = run_evaluation(data, all_sequences)
    logger.info(f"Metrics: {metrics_agg}")

    # Create metrics plot
    create_metrics_plot(metrics_list, output_dir)

    # Print statistics
    if metrics_list:
        avg_retrievals = sum(m['num_retrieval'] for m in metrics_list) / len(metrics_list)
        logger.info(f"Average retrievals per case: {avg_retrievals:.2f}")
    else:
        logger.info("No cases to compute average retrievals.")

    # Compute token statistics
    output_token_counts = []
    for seq in all_sequences:
        if tokenizer is not None:
            tokens = tokenizer.encode(seq["output"])
            output_token_counts.append(len(tokens))
    
    if output_token_counts:
        avg_output_tokens = sum(output_token_counts) / len(output_token_counts)
        logger.info(f"Average output tokens per case: {avg_output_tokens:.2f}")
    else:
        logger.info("No outputs to compute average token count.")

    # Save all outputs
    save_outputs(data, all_sequences, metrics_list, retrieval_history, metrics_agg, output_dir)

if __name__ == '__main__':
    main()