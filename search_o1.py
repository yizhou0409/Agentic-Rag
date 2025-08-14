#!/usr/bin/env python3
"""
Search-o1: Agentic Search-Enhanced Large Reasoning Models

This implementation reproduces the logic from the Search-o1 paper:
https://arxiv.org/abs/2501.05366

The system consists of three main components:
1. Reasoner: LLM that performs reasoning and decides when to search
2. Retriever: Retrieves relevant documents based on search queries
3. Summarizer: Summarizes retrieved documents for the reasoner
"""

import os
# Set JAX to use CPU backend to avoid CUDA/cuDNN initialization issues
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['JAX_ENABLE_X64'] = 'True'
# Disable GPU for JAX to prevent cuDNN errors
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import json
import argparse
import re
import yaml
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import logging



# Import local modules
from utils import calculate_metrics

# Import retriever modules
from e5_retriever import E5Retriever
from bm25_retriever import BM25Retriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StopOnStringCriteria(StoppingCriteria):
    """Stopping criteria that stops generation when a specific string is found in newly generated tokens."""
    
    def __init__(self, tokenizer, stop_strings, initial_length):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings if isinstance(stop_strings, list) else [stop_strings]
        self.initial_length = initial_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check the newly generated tokens (after the initial input)
        if input_ids.shape[1] <= self.initial_length:
            return False
        
        # Decode only the newly generated part
        new_tokens = input_ids[0][self.initial_length:]
        decoded_new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return any(stop_string in decoded_new_text for stop_string in self.stop_strings)








@dataclass
class SearchO1Config:
    """Configuration for Search-o1 system."""
    # Model settings
    reasoner_model_name: str = "Qwen/Qwen3-32B"
    summarizer_model_name: str = "Qwen/Qwen3-32B"
    retriever_type: str = "bm25"  # or "e5"
    retriever_index_path: str = "indexes/bm25"  # Only used for bm25 and e5
    e5_model_path: str = "intfloat/e5-large-v2"  # Path to E5 model
    
    # Generation settings
    max_turns: int = 5
    max_new_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    
    # Retrieval settings
    top_k_docs: int = 10
    
    # Dataset settings
    dataset_name: str = "hotpotqa"  # or "2wikimultihop"
    max_samples: Optional[int] = None
    
    # Output settings
    output_dir: str = "output/search_o1"
    save_intermediate: bool = True


class Reasoner:
    """The main reasoning model that decides when to search and provides answers."""
    
    def __init__(self, model_name: str, config: SearchO1Config):
        self.model_name = model_name
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
        self._load_prompt_template()
    
    def _load_model(self):
        """Load the Qwen3 model with recommended settings."""
        logger.info(f"Loading reasoner model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        logger.info("Reasoner model loaded successfully")
    
    def _load_prompt_template(self):
        """Load the prompt template for the reasoner."""
        prompt_path = "prompts/default_QA.yaml"
        with open(prompt_path, 'r') as f:
            prompt_data = yaml.safe_load(f)
        
        self.prompt_template = prompt_data['user_prompt']
    
    def generate_response(self, sequence: str) -> str:
        """
        Generate a response from the reasoner with stop words for <search> and <answer> tags.
        
        Args:
            question: The question to answer
            sequence: Optional sequence containing previous responses and information.
                     If None, initializes a new sequence with the question under prompt template.
            
        Returns:
            Generated response text
        """
        
        # Tokenize input
        model_inputs = self.tokenizer([sequence], return_tensors="pt").to(self.model.device)
        initial_length = model_inputs['input_ids'].shape[1]
        
        # Create stopping criteria for </search> and </answer> tags
        # We want to stop after the closing tags, not before them
        stopping_criteria = StoppingCriteriaList([
            StopOnStringCriteria(self.tokenizer, ["</search>", "</answer>"], initial_length)
        ])
        
        # Generate response with proper stopping criteria
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                min_p=self.config.min_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(generated_ids[0][initial_length:], skip_special_tokens=True)
        
        return generated_text


class Summarizer:
    """Summarizes retrieved documents for the reasoner."""
    
    def __init__(self, model_name: str, config: SearchO1Config):
        self.model_name = model_name
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
        self._load_prompt_template()
    
    def _load_model(self):
        """Load the summarizer model."""
        logger.info(f"Loading summarizer model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        logger.info("Summarizer model loaded successfully")
    
    def _load_prompt_template(self):
        """Load the prompt template for summarization."""
        prompt_path = "prompts/default_retrieval_summary.yaml"
        with open(prompt_path, 'r') as f:
            prompt_data = yaml.safe_load(f)
        
        self.prompt_template = prompt_data['user_prompt']
    
    def summarize_documents(self, question: str, documents: List[str]) -> str:
        """
        Summarize retrieved documents for the given question.
        
        Args:
            question: The original question
            documents: List of retrieved document texts
            
        Returns:
            Summarized information
        """
        # Combine documents
        combined_docs = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])
        
        # Prepare prompt
        prompt = self.prompt_template.format(question=question, documents=combined_docs)
        
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.3,  # Lower temperature for more focused summaries
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the input)
        summary = generated_text[len(text):]
        
        # Extract the final summary from the generated text
        # The prompt expects output to start with "### Extracted Information"
        if "### Extracted Information" in summary:
            # Extract everything after "### Extracted Information"
            start_idx = summary.find("### Extracted Information")
            if start_idx != -1:
                # Get the text after the header
                extracted_info = summary[start_idx + len("### Extracted Information"):].strip()
                # Remove any trailing markdown or extra text
                if "\n\n" in extracted_info:
                    extracted_info = extracted_info.split("\n\n")[0]
                if "\n###" in extracted_info:
                    extracted_info = extracted_info.split("\n###")[0]
                return extracted_info.strip()
        
        # If the expected format is not found, return the full summary
        return summary.strip()
    



class SearchO1System:
    """Main Search-o1 system that coordinates reasoner, retriever, and summarizer."""
    
    def __init__(self, config: SearchO1Config):
        self.config = config
        self.reasoner = Reasoner(config.reasoner_model_name, config)
        self.summarizer = Summarizer(config.summarizer_model_name, config)
        self.retriever = self._load_retriever()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _load_retriever(self):
        """Load the appropriate retriever based on configuration."""
        if self.config.retriever_type == "bm25":
            return BM25Retriever(self.config.retriever_index_path, self.config.top_k_docs)
        elif self.config.retriever_type == "e5":
            return E5Retriever(self.config.retriever_index_path, self.config.e5_model_path)
        else:
            raise ValueError(f"Unsupported retriever type: {self.config.retriever_type}")
    
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from <search> tags."""
        search_pattern = r'<search>(.*?)</search>'
        match = re.search(search_pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from <answer> tags."""
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    

    
    def process_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Process an entire dataset using sequence-based processing.
        Each question maintains its own sequence of responses and information.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            List of results for each question
        """
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Load dataset
        questions = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if self.config.dataset_name == "hotpotqa":
                    questions.append({
                        "id": data["id"],
                        "question": data["question"],
                        "golden_answers": data["golden_answers"]
                    })
                else:  # 2wikimultihop
                    questions.append({
                        "id": data["_id"],
                        "question": data["text"],
                        "golden_answers": data["metadata"]["answer"]
                    })
        
        logger.info(f"Loaded {len(questions)} questions from dataset")
        
        # Limit samples if specified
        if self.config.max_samples:
            original_count = len(questions)
            questions = questions[:self.config.max_samples]
            logger.info(f"Limited to {len(questions)} questions (max_samples: {self.config.max_samples}, original: {original_count})")
        
        logger.info(f"Processing {len(questions)} questions")
        
        # Initialize all questions with their sequences
        active_questions = []
        for question_data in questions:
            # Initialize sequence with the question under prompt template
            prompted_question = self.reasoner.prompt_template.format(question=question_data["question"])
            
            # Prepare messages for chat template
            messages = [{"role": "user", "content": prompted_question}]
            initial_sequence = self.reasoner.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            active_questions.append({
                "id": question_data["id"],
                "question": question_data["question"],
                "golden_answers": question_data["golden_answers"],
                "sequence": initial_sequence,  # Initialize sequence with question
                "response": "",
                "turns": [],
                "final_turn": 0,
                "answer": None,
                "error": None
            })
        
        max_turn_warning = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."
        completed_questions = []
        # Process all questions together in turns
        for turn_num in range(self.config.max_turns + 1):
            logger.info(f"Processing Turn {turn_num + 1} with {len(active_questions)} active questions")
            logger.info(f"Remaining active questions: {len(active_questions)}, Completed questions: {len(completed_questions)}")
            
            if not active_questions:
                break
            
            # Step 1: Generate responses for all active questions
            search_queries = {}  # question_id -> search_query
            
            for question_data in tqdm(active_questions, desc="Processing questions"):
                # Generate response using the current sequence
                response = self.reasoner.generate_response(
                    question_data["sequence"]
                )
                
                # Extract search query or answer
                search_query = self._extract_search_query(response)
                answer = self._extract_answer(response)
                
                # Record turn info
                turn_info = {
                    "turn": turn_num + 1,
                    "response": response,
                    "search_query": search_query,
                    "answer": answer
                }
                question_data["turns"].append(turn_info)
                
                # Update sequence with the current response
                question_data["sequence"] += response
                question_data["response"] += response
                
                if answer:
                    # Question answered, mark as completed
                    question_data["answer"] = answer
                    question_data["final_turn"] = turn_num + 1
                    completed_questions.append(question_data)
                    logger.info(f"Question {question_data['id']} answered in turn {turn_num + 1}")
                elif search_query:
                    # Need to search, collect search query
                    search_queries[question_data["id"]] = search_query
                else:
                    # No search query or answer found
                    question_data["error"] = "No search query or answer found"
                    question_data["final_turn"] = turn_num + 1
                    completed_questions.append(question_data)
                    logger.warning(f"Question {question_data['id']}: No search query or answer found")
            
            # Step 2: Remove completed questions from active list
            active_questions = [q for q in active_questions if q["id"] not in [cq["id"] for cq in completed_questions]]

            
            # Step 3: Process all search queries together (if any)
            if search_queries:
                logger.info(f"Processing {len(search_queries)} search queries in turn {turn_num + 1}")
                
                # Collect all unique search queries to avoid duplicate retrievals
                unique_queries = list(set(search_queries.values()))
                query_to_questions = {}
                
                for question_id, query in search_queries.items():
                    if query not in query_to_questions:
                        query_to_questions[query] = []
                    query_to_questions[query].append(question_id)
                
                # Process each unique query
                for query in tqdm(unique_queries, desc="Processing search queries"):
                    try:
                        # Retrieve documents
                        retrieved_docs = self.retriever.search(query, num=self.config.top_k_docs)
                        
                        doc_texts = []
                        for doc in retrieved_docs:
                            if isinstance(doc, dict):
                                doc_text = doc.get('text', doc.get('contents', str(doc)))
                            else:
                                doc_text = str(doc)
                            doc_texts.append(doc_text)
                        
                        # Summarize documents
                        summary = self.summarizer.summarize_documents(query, doc_texts)
                        
                        # Update sequence for all questions that used this query
                        for question_id in query_to_questions[query]:
                            for question_data in active_questions:
                                if question_data["id"] == question_id:
                                    # Append information to the sequence with proper formatting
                                    information_block = f"<information> {summary} </information>"
                                    question_data["sequence"] += information_block
                                    question_data["response"] += information_block
                                    
                                    # Update turn info with retrieval results (but don't include in final results)
                                    for turn_info in question_data["turns"]:
                                        if turn_info["search_query"] == query:
                                            turn_info["retrieved_docs"] = [str(doc) for doc in retrieved_docs]
                                            turn_info["summary"] = summary
                                            break
                                    break
                    except Exception as e:
                        logger.error(f"Error processing query '{query}': {e}")
                        # Continue with other queries
            
            # Step 4: Check if max turns reached for remaining questions
            if turn_num == self.config.max_turns - 1 and active_questions:
                for question_data in active_questions:
                    question_data["sequence"] += max_turn_warning
                
            elif turn_num == self.config.max_turns:
                for question_data in active_questions:
                    question_data["error"] = "Max turns reached"
                    question_data["final_turn"] = self.config.max_turns
                    completed_questions.append(question_data)
                active_questions = []
        
        # Combine all completed questions
        all_results = completed_questions
        
        # Calculate metrics for all results
        for result in all_results:
            if result["answer"]:
                metrics = calculate_metrics(result["answer"], result["golden_answers"])
                result["metrics"] = metrics
            else:
                result["metrics"] = {"em": 0.0, "f1": 0.0}
            
        # Save intermediate results
        if self.config.save_intermediate:
            self._save_intermediate_results(all_results)
        
        logger.info(f"Processed {len(all_results)} questions successfully")
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to file."""
        output_file = os.path.join(self.config.output_dir, "intermediate_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def save_final_results(self, results: List[Dict[str, Any]], dataset_name: str):
        """Save final results and evaluation metrics."""
        # Prepare results without retrieved documents for main results file
        clean_results = []
        retrieval_results = []
        
        for result in results:
            # Create clean result without retrieved documents
            clean_result = result.copy()
            
            # Remove retrieved_docs from turns for main results
            for turn in clean_result["turns"]:
                if "retrieved_docs" in turn:
                    # Store retrieval info separately
                    retrieval_results.append({
                        "question_id": result["id"],
                        "turn": turn["turn"],
                        "search_query": turn["search_query"],
                        "retrieved_docs": turn["retrieved_docs"],
                        "summary": turn.get("summary", "")
                    })
                    # Remove from main results
                    del turn["retrieved_docs"]
                    if "summary" in turn:
                        del turn["summary"]
            
            clean_results.append(clean_result)
        
        # Save detailed results (without retrieved documents)
        output_file = os.path.join(self.config.output_dir, f"{dataset_name}_results.json")
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        # Save retrieval results separately
        if retrieval_results:
            retrieval_file = os.path.join(self.config.output_dir, f"{dataset_name}_retrieval_results.json")
            with open(retrieval_file, 'w') as f:
                json.dump(retrieval_results, f, indent=2)
            logger.info(f"Retrieval results saved to {retrieval_file}")
        
        # Calculate and save summary metrics
        total_questions = len(clean_results)
        answered_questions = sum(1 for r in clean_results if r["answer"] is not None)
        
        avg_em = sum(r["metrics"]["em"] for r in clean_results) / total_questions
        avg_f1 = sum(r["metrics"]["f1"] for r in clean_results) / total_questions
        
        avg_turns = sum(r["final_turn"] for r in clean_results) / total_questions
        
        summary = {
            "dataset": dataset_name,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "answer_rate": answered_questions / total_questions,
            "average_em": avg_em,
            "average_f1": avg_f1,
            "average_turns": avg_turns,
            "config": self.config.__dict__
        }
        
        summary_file = os.path.join(self.config.output_dir, f"{dataset_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"Average EM: {avg_em:.3f}, Average F1: {avg_f1:.3f}")


def main():
    """Main function to run Search-o1 system."""
    parser = argparse.ArgumentParser(description="Search-o1: Agentic Search-Enhanced Large Reasoning Models")
    
    # Model settings
    parser.add_argument("--reasoner-model", default="Qwen/Qwen3-32B", help="Reasoner model name")
    parser.add_argument("--summarizer-model", default="Qwen/Qwen3-32B", help="Summarizer model name")
    parser.add_argument("--retriever-type", default="bm25", choices=["bm25", "e5"], help="Retriever type")
    parser.add_argument("--retriever-index-path", default="indexes/bm25", help="Path to retriever index")
    parser.add_argument("--e5-model-path", default="intfloat/e5-large-v2", help="Path to E5 model for retrieval")
    
    # Generation settings
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum number of turns")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-p sampling")
    
    # Retrieval settings
    parser.add_argument("--top-k-docs", type=int, default=10, help="Number of documents to retrieve")
    
    # Dataset settings
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimultihop"], help="Dataset to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    
    # Output settings
    parser.add_argument("--output-dir", default="output/search_o1", help="Output directory")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    
    args = parser.parse_args()
    
    # Create config
    config = SearchO1Config(
        reasoner_model_name=args.reasoner_model,
        summarizer_model_name=args.summarizer_model,
        retriever_type=args.retriever_type,
        retriever_index_path=args.retriever_index_path,
        e5_model_path=args.e5_model_path,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        top_k_docs=args.top_k_docs,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate
    )
    
    # Create system
    system = SearchO1System(config)
    
    # Determine dataset path
    if args.dataset == "hotpotqa":
        dataset_path = "data/hotpotqa/dev.jsonl"
    else:  # 2wikimultihop
        dataset_path = "data/2wikimultihop/queries.jsonl"
    
    # Process dataset
    results = system.process_dataset(dataset_path)
    
    # Save results
    system.save_final_results(results, args.dataset)
    
    logger.info("Search-o1 processing completed!")


if __name__ == "__main__":
    main()
