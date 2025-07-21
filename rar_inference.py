import os
# Set JAX to use CUDA GPU backend specifically (avoiding ROCm initialization issues)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['JAX_ENABLE_X64'] = 'True'

import flashrag.retriever  # HACK: although not used, importing this before importing load_dataset somehow make sure get_retriever() can work. Dependency bug of flashrag/faiss/datasets.
from datasets import load_dataset
from transformers import AutoTokenizer
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

logger = logging.getLogger(__name__)

# Constants for XML-like tags
BEGIN_OF_THINK, END_OF_THINK = "<think>", "</think>"
BEGIN_OF_ANSWER, END_OF_ANSWER = "<answer>", "</answer>"
BEGIN_OF_SEARCH, END_OF_SEARCH = "<search>", "</search>"
BEGIN_OF_INFO, END_OF_INFO = "<information>", "</information>"

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
    reasoner_tokenizer: Optional[AutoTokenizer]
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
        
        if reasoner_tokenizer is not None:
            prompt = reasoner_tokenizer.apply_chat_template(
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
    summarizer_tokenizer: Optional[AutoTokenizer]
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
        
        if summarizer_tokenizer is not None:
            prompt = summarizer_tokenizer.apply_chat_template(
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
    summarizer_tokenizer: Optional[AutoTokenizer],
    use_openai_summarizer: bool
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
            
            if summarizer_tokenizer is not None:
                prompt = summarizer_tokenizer.apply_chat_template(
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
        
        if summarizer_tokenizer is not None:
            agg_prompt = summarizer_tokenizer.apply_chat_template(
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
        
        retrieval_history_entries.append({
            "query": query,
            "per_doc_summaries": per_doc_summaries,
            "per_doc_raw_outputs": [out["text"] for out in per_doc_outputs_raw],
            "final_summary": final_summary,
            "final_summary_raw_output": final_summary_output_raw['text'],
            "prompt_and_output": (
                agg_prompt if summarizer_tokenizer is None 
                else agg_prompt + "\n\n[OUTPUT STARTS HERE]\n\n" + final_analysis + "\n\n" + final_summary
            )
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
    
    reasoner_tokenizer = None if use_openai_reasoner else AutoTokenizer.from_pretrained(cfg.model.reasoner_name_or_path)
    reasoner_model = GeneratorMixin(
        mode=cfg.model.reasoner_mode,
        server_url=cfg.model.reasoner_url,
        model_name_or_path=cfg.model.reasoner_name_or_path,
        server_params=OmegaConf.to_container(cfg.inference.server_params),
    )

    summarizer_tokenizer = None if use_openai_summarizer else AutoTokenizer.from_pretrained(cfg.model.summarizer_name_or_path)
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
    tokenizer = reasoner_tokenizer if not use_openai_reasoner else None
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
    if reasoner_tokenizer is not None:
        reasoner_sampling_params["stop"] = [END_OF_SEARCH, END_OF_ANSWER, reasoner_tokenizer.eos_token]
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

        # Process reasoner outputs
        search_queries = []
        for seq, out in zip(active_sequences, reasoner_outputs):
            generated = out["text"]
            seq["output"] += generated
            thought, search_q, answer = parse_reasoning_generation(generated)
            
            if search_q:
                search_queries.append(search_q)
                if use_openai_reasoner:
                    seq["messages"].append({"role": "assistant", "content": generated.rstrip()})
                else:
                    seq["prompt"] += generated.rstrip()
            elif answer:
                seq["finished"] = True
                seq["answer"] = answer
            else:
                seq["finished"] = True
                seq["answer"] = ""
        
        active_sequences = [seq for seq in active_sequences if not seq["finished"]]

        # Retrieval and summarization
        if search_queries:
            start_search_time = time.time()
            search_results = retriever.batch_search(search_queries)
            ret_time = time.time() - start_search_time
            logger.info(f"Retrieval time: {ret_time:.2f} seconds")

            # Try simple summarization first
            summarizer_prompt = build_summarizer_prompt(
                search_queries, search_results, summarizer_user_message_template, 
                None, cfg, summarizer_tokenizer
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
                for seq, out in zip(active_sequences, summary_outputs):
                    summary = out[1].strip()
                    append_summary = f"{BEGIN_OF_INFO} {summary} {END_OF_INFO}\n\n"
                    if use_openai_reasoner:
                        seq["messages"].append({"role": "user", "content": append_summary})
                    else:
                        seq["prompt"] += append_summary
                    seq["output"] += append_summary
            else:
                # Fall back to per-document summarization
                retrieval_history_entries = process_per_document_summarization(
                    search_queries, search_results, summarizer_user_message_template,
                    summarizer_model, summarizer_sampling_params, summarizer_tokenizer, use_openai_summarizer
                )
                retrieval_history.extend(retrieval_history_entries)
                
                for seq, entry in zip(active_sequences, retrieval_history_entries):
                    summary = entry["final_summary"].strip()
                    append_summary = f"{BEGIN_OF_INFO} {summary} {END_OF_INFO}\n\n"
                    if use_openai_reasoner:
                        seq["messages"].append({"role": "user", "content": append_summary})
                    else:
                        seq["prompt"] += append_summary
                    seq["output"] += append_summary

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
        if reasoner_tokenizer is not None:
            tokens = reasoner_tokenizer.encode(seq["output"])
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