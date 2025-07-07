import os
# Set JAX to use CUDA GPU backend specifically (avoiding ROCm initialization issues)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['JAX_ENABLE_X64'] = 'True'

import flashrag.retriever # HACK: although not used, importing this before importing load_dataset somehow make sure get_retriever() can work. Dependency bug of flashrag/faiss/datasets.
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
import asyncio
import requests

import time
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


BEGIN_OF_THINK, END_OF_THINK = "<think>", "</think>"
BEGIN_OF_ANSWER, END_OF_ANSWER = "<answer>", "</answer>"
BEGIN_OF_SEARCH, END_OF_SEARCH = "<search>", "</search>"
BEGIN_OF_INFO, END_OF_INFO = "<information>", "</information>"

# --- API-call model support ---
# Add use_openai_format flag and logic from reference_api

def format_initial_qa_prompt(item, user_message_template, sys_message, tokenizer, use_openai_format=False):
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
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt}

def format_retrieval_summary_prompts(search_queries, search_results, user_message_template, sys_message, tokenizer, use_openai_format=False):
    def format_doc(doc):
        title = doc.get('title', doc.get('id', 'Unknown'))
        text = doc.get('text', doc.get('content', str(doc)))
        return f"Document (Title: {title}): {text}"
    prompts = []
    for query, results in zip(search_queries, search_results):
        retrieved_context = "\n\n".join([format_doc(result) for result in results])
        user_message = user_message_template.format(question=query, documents=retrieved_context)
        sys_content = sys_message if sys_message is not None else "You are a helpful assistant."
        user_content = user_message if user_message is not None else ""
        if use_openai_format:
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ]
            prompts.append(messages)
        else:
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
    return prompts

# --- Main function update ---
@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg):
    logger.info("***Experiment Config***")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("***********************")

    dataset_path = os.path.join(cfg.data.dataset_root_dir, cfg.data.dataset_name, f"{cfg.data.dataset_split}.jsonl")
    data = load_dataset("json", data_files=dataset_path, split="train") # not the actual split

    # Limit to only 10 cases for debugging or speed
    data = data.select(range(min(10, len(data))))

    if cfg.data.subset_size:
        data = data.shuffle(seed=42).select(range(cfg.data.subset_size))
        logger.info(f"Using a subset of the data: {cfg.data.subset_size}")
    
    # Load tokenizers for reasoner and summarizer separately
    reasoner_tokenizer = AutoTokenizer.from_pretrained(cfg.model.reasoner_name_or_path)
    summarizer_tokenizer = AutoTokenizer.from_pretrained(cfg.model.summarizer_name_or_path)
    
    reasoner_sys_message = load_sys_message(cfg.model.reasoner_name_or_path)
    reasoner_user_message_template = load_user_message_template(cfg.inference.prompt_templates_dir, cfg.inference.user_message_template_name)
    # Detect API-call model usage
    use_openai_format = cfg.model.reasoner_mode == "openai" or cfg.model.summarizer_mode == "openai"
    if not use_openai_format:
        tokenizer = reasoner_tokenizer
    else:
        tokenizer = None
    data = data.map(lambda x: format_initial_qa_prompt(x, reasoner_user_message_template, reasoner_sys_message, tokenizer, use_openai_format), batched=False)

    summarizer_sys_message = load_sys_message(cfg.model.summarizer_name_or_path)
    summarizer_user_message_template = load_user_message_template(cfg.inference.prompt_templates_dir, "default_retrieval_summary")
    if use_openai_format:
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
   
    logger.info(f"Example Prompt:")
    logger.info(all_sequences[0]["prompt"])

    # Define the output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Initialize the LLM
    server_params = OmegaConf.to_container(cfg.inference.server_params)
    reasoner_sampling_params = OmegaConf.to_container(cfg.inference.reasoner_sampling_params)
    reasoner_sampling_params["stop"] = [END_OF_SEARCH, END_OF_ANSWER, reasoner_tokenizer.eos_token]
    summarizer_sampling_params = OmegaConf.to_container(cfg.inference.summarizer_sampling_params)
    
    reasoner = GeneratorMixin(
        mode = cfg.model.reasoner_mode,
        server_url = cfg.model.reasoner_url,
        model_name_or_path = cfg.model.reasoner_name_or_path,
        server_params = server_params,
    )
    
    summarizer = GeneratorMixin(
        mode = cfg.model.summarizer_mode,
        server_url = cfg.model.summarizer_url,
        model_name_or_path = cfg.model.summarizer_name_or_path,
        server_params = server_params,
    )
    
    # Initialize retreiver model and index
    retriever_config = Config(config_dict=OmegaConf.to_container(cfg.retrieval.flashrag_config))
    retriever = get_retriever(retriever_config)

    retrieval_history = []
    
    # Load the full corpus for index lookup
    corpus_path = cfg.retrieval.flashrag_config.corpus_path
    with open(corpus_path, "r") as f:
        full_corpus = [json.loads(line) for line in f]
    
    # Ensure each data item has a 'gold_answer' field for evaluation compatibility
    for item in data:
        if 'gold_answer' not in item and 'golden_answers' in item:
            item['gold_answer'] = item['golden_answers']

    # Main Inference Loop
    turn = 0
    active_sequences = all_sequences
    while turn < cfg.inference.max_turns:
        if not active_sequences:
            break
        
        turn += 1
        print(f'\n\n-------------- Turn {turn} --------------')
        print(f'# Active Sequences: {len(active_sequences)}')

        # Generate responses
        prompts = [seq["prompt"] for seq in active_sequences]
        start_time = time.time()
        outputs = reasoner.generate(prompts, sampling_params=reasoner_sampling_params)
        end_time = time.time()
        gen_time = end_time - start_time
        logger.info(f"Reasoning Generation time: {gen_time:.2f} seconds")
        print(f"[Turn {turn}] Generation time: {gen_time:.2f} seconds")

        search_queries = []

        for seq, out in zip(active_sequences, outputs):
            generated = out["text"]
            
            seq["output"] += generated
            thought, search_q, answer = parse_reasoning_generation(generated)

            if search_q:
                search_queries.append(search_q)
                seq["prompt"] += generated.rstrip() # generation sometimes ends on </search>\n\n\n with arbitrary number of newlines. We add two newlines before the <information> token in later code.
            elif answer:
                seq["finished"] = True
                seq["answer"] = answer
            else:
                # invalid
                seq["finished"] = True
                seq["answer"] = ""
        
        active_sequences = [seq for seq in active_sequences if not seq["finished"]]
        
        # Perform search and summarize
        if search_queries:
            start_search_time = time.time()
            search_results = retriever.batch_search(search_queries)
            end_search_time = time.time()
            ret_time = end_search_time - start_search_time
            logger.info(f"Retrieval time: {ret_time:.2f} seconds")
            print(f"[Turn {turn}] Retrieval time: {ret_time:.2f} seconds")

            per_doc_summary_template = summarizer_user_message_template  # Use the same template for per-doc
            per_doc_summaries_all = []
            final_summary_outputs = []
            retrieval_history_entries = []

            for query, results in zip(search_queries, search_results):
                # 1. Summarize each document individually
                per_doc_prompts = []
                doc_titles = []
                for doc in results:
                    # Format single doc as context
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
                    doc_titles.append(doc_title)
                    user_message = per_doc_summary_template.format(question=query, documents=f"Document (Title: {doc_title}): {doc_text}")
                    messages = [
                        {"role": "system", "content": summarizer_sys_message},
                        {"role": "user", "content": user_message},
                    ]
                    prompt = summarizer_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    per_doc_prompts.append(prompt)
                # Generate per-document summaries
                per_doc_outputs_raw = summarizer.generate(per_doc_prompts, summarizer_sampling_params)
                per_doc_summaries = [parse_summary_generation(out["text"])[1] for out in per_doc_outputs_raw]

                # 2. Aggregate all per-document summaries and generate a final summary
                summaries_context = "\n\n".join([f"Summary for Document {i+1} (Title: {doc_titles[i]}): {per_doc_summaries[i]}" for i in range(len(per_doc_summaries))])
                agg_user_message = (
                    "Given the following summaries of retrieved documents, write a concise, factual, and relevant answer to the user query.\n"
                    "You must only use information from the summaries.\n"
                    "\n"
                    "### User Query\n"
                    f"{query}\n"
                    "\n"
                    "### Document Summaries\n"
                    f"{summaries_context}"
                )
                agg_messages = [
                    {"role": "system", "content": summarizer_sys_message},
                    {"role": "user", "content": agg_user_message},
                ]
                agg_prompt = summarizer_tokenizer.apply_chat_template(
                    agg_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                final_summary_output_raw = summarizer.generate([agg_prompt], summarizer_sampling_params)[0]
                final_analysis, final_summary = parse_summary_generation(final_summary_output_raw["text"])

                # 3. Save both per-document summaries and the final summary
                retrieval_history_entries.append({
                    "query": query,
                    "per_doc_summaries": per_doc_summaries,
                    "per_doc_raw_outputs": [out["text"] for out in per_doc_outputs_raw],
                    "final_summary": final_summary,
                    "final_summary_raw_output": final_summary_output_raw["text"],
                    "prompt_and_output": agg_prompt + "\n\n[OUTPUT STARTS HERE]\n\n" + final_analysis + "\n\n" + final_summary
                })

            retrieval_history.extend(retrieval_history_entries)

            # Update input prompts for the next turn (use the final summary)
            for seq, entry in zip(active_sequences, retrieval_history_entries):
                summary = entry["final_summary"].strip()
                append_summary = f"{BEGIN_OF_INFO} {summary} {END_OF_INFO}\n\n" # not anding the <think> token here
                seq["prompt"] += append_summary
                seq["output"] += append_summary

    # if there are still active sequences, we force them to finish by adding a short thought
    print("-------- Turn limit reached. (Soft) Forcing active sequences to finish. --------")
    for seq in active_sequences:
        timeout_message = f"Time is up.\n\nSince I have searched for {cfg.inference.max_turns} times, I am not allowed to search anymore. I should give a final answer now with the information I have."
        seq["prompt"] += timeout_message
        seq["output"] += timeout_message
    prompts = [seq["prompt"] for seq in active_sequences]
    # outputs = llm.generate(prompts, sampling_params=sampling_params)
    # outputs = requests.post(
    #     cfg.model.llm_client_url+"/generate",
    #     json={"text": prompts, "sampling_params": llm_sampling_params},
    # ).json()
    outputs = reasoner.generate(prompts, reasoner_sampling_params)
    for seq, out in zip(active_sequences, outputs):
        generated = out["text"]
        seq["output"] += generated
        thought, search_q, answer = parse_reasoning_generation(generated)
        if answer:
            seq["finished"] = True
            seq["answer"] = answer
        else:
            seq["finished"] = True
            seq["answer"] = ""

    reasoner.shutdown()
    summarizer.shutdown()

    # Evaluate the generated answers
    metrics_list, metrics_agg = run_evaluation(data, all_sequences)
    logger.info(f"Metrics: {metrics_agg}")
    
    # --- New: Analyze retrieval counts and metrics by retrieval times ---
    retrieval_to_metrics = defaultdict(list)
    for m in metrics_list:
        retrieval_to_metrics[m['num_retrieval']].append(m)

    # Compute average metrics for each retrieval count
    retrieval_counts = sorted(retrieval_to_metrics.keys())
    avg_em = []
    avg_cover = []
    avg_f1 = []
    for rc in retrieval_counts:
        ms = retrieval_to_metrics[rc]
        avg_em.append(sum(x['em'] for x in ms) / len(ms))
        avg_cover.append(sum(x['cover_match'] for x in ms) / len(ms))
        avg_f1.append(sum(x['str_f1'] for x in ms) / len(ms))

    # Plot
    plt.figure(figsize=(8,6))
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
    print(f"Saved retrieval-metrics plot to {plot_path}")
    
    # --- New: Print average retrievals per case ---
    if metrics_list and len(metrics_list) > 0:
        avg_retrievals = sum(m['num_retrieval'] for m in metrics_list) / len(metrics_list)
        print(f"Average retrievals per case: {avg_retrievals:.2f}")
    else:
        print("No cases to compute average retrievals.")
    
    # --- Compute and print average output tokens ---
    output_token_counts = []
    for seq in all_sequences:
        tokens = reasoner_tokenizer.encode(seq["output"])
        output_token_counts.append(len(tokens))
    if output_token_counts:
        avg_output_tokens = sum(output_token_counts) / len(output_token_counts)
        print(f"Average output tokens per case: {avg_output_tokens:.2f}")
    else:
        print("No outputs to compute average token count.")
    
    # save outputs
    t = datetime.datetime.now()
    outputs = []
    for item, seq, metrics in zip(data, all_sequences, metrics_list):
        outputs.append({
            "prompt": seq["prompt"],
            "output": seq["output"],
            "question": item["question"],
            "pred": seq["answer"],
            "gold": item["golden_answers"],
            "finished": seq["finished"],
            "metrics": metrics,
        })
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_file_name = f'output_{t.strftime("%m-%d-%H-%M-%S")}.json'
    with open(f"{output_dir}/{output_file_name}", "w") as f:
        json.dump(outputs, f, indent=2)

    # save retrieval history
    retrieval_history_file_name = f'retrieval_history_{t.strftime("%m-%d-%H-%M-%S")}.json'
    with open(f"{output_dir}/{retrieval_history_file_name}", "w") as f:
        json.dump(retrieval_history, f, indent=2)
    
    # save metrics 
    metrics_file_name = f'metrics_{t.strftime("%m-%d-%H-%M-%S")}.json'
    with open(f"{output_dir}/{metrics_file_name}", "w") as f:
        json.dump(metrics_agg, f, indent=2)

if __name__ == '__main__':
    main()