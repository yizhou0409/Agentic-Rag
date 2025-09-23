#!/usr/bin/env python3
"""
Inference System for Agentic RAG
The system consists of three main components:
1. Reasoner: LLM that performs reasoning and decides when to search
2. Retriever: Retrieves relevant documents based on search queries
3. Summarizer: Summarizes retrieved documents for the reasoner
"""

import os
import json
import argparse
import re
import yaml
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from tqdm import tqdm
import logging

# Import local modules
from utils import calculate_metrics

# Import retriever modules
from e5_retriever import E5Retriever
from bm25_retriever import BM25Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StopOnStringCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, initial_length):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings if isinstance(stop_strings, list) else [stop_strings]
        self.initial_length = initial_length

    def __call__(self, input_ids, scores):
        # Check if any of the stop strings are in newly generated tokens
        new_tokens = input_ids[0][self.initial_length:]
        decoded_new_tokens = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return any(stop_string in decoded_new_tokens for stop_string in self.stop_strings)

@dataclass
class InferenceConfig:
    # Model settings
    reasoner_model_name: str = "Qwen/Qwen3-32B"
    summarizer_model_name: str = "Qwen/Qwen3-32B"
    reasoner_lora_path: Optional[str] = None  # Path to LoRA adapters for reasoner
    summarizer_lora_path: Optional[str] = None  # Path to LoRA adapters for summarizer
    retriever_type: str = "bm25"  # or "e5"
    retriever_index_path: str = "indexes/bm25"
    e5_model_path: str = "intfloat/e5-large-v2"
    # Generation settings
    max_turns: int = 5
    max_new_tokens: int = 2048
    greedy_thinking: bool = False  # Use greedy decoding in thinking mode
    high_randomness_mode: bool = False  # Use more aggressive random generation for diverse trajectories
    # Retrieval settings
    top_k_docs: int = 10
    # Dataset settings
    dataset_name: str = "hotpotqa"
    split: str = "dev"  # train or dev
    max_samples: Optional[int] = None
    output_dir: str = "output/search_o1"
    save_intermediate: bool = False
    start_sample: Optional[int] = None  # 1-based inclusive
    end_sample: Optional[int] = None    # 1-based inclusive
    # Probe settings
    use_probe: bool = False
    probe_path: str = "probe/"
    probe_confidence_threshold: float = 0.7

class Reasoner:
    """The main reasoning model that decides when to search and provides answers."""
    def __init__(self, model_name: str, config: InferenceConfig):
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
        
        # Load LoRA adapters if specified
        if self.config.reasoner_lora_path:
            logger.info(f"Loading LoRA adapters from: {self.config.reasoner_lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.reasoner_lora_path)
            logger.info("LoRA adapters loaded successfully")
        
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
            # Use greedy decoding if specified, otherwise use sampling for thinking mode
            if self.config.greedy_thinking:
                logger.debug("Using greedy decoding for thinking mode")
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria
                )
            elif self.config.high_randomness_mode:
                # Use more aggressive random settings for diverse trajectory generation (DPO training)
                logger.debug("Using high randomness mode for diverse trajectory generation")
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=1.2,  # Higher temperature for more randomness
                    top_p=0.9,        # Higher top_p for more diversity
                    top_k=40,         # Higher top_k for more token options
                    min_p=0.05,       # Minimum probability threshold
                    repetition_penalty=1.1,  # Slight repetition penalty to avoid loops
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria
                )
            else:
                logger.debug("Using standard sampling-based decoding for thinking mode")
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.6,  # thinking mode
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
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
    def __init__(self, model_name: str, config: InferenceConfig):
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
        
        # Load LoRA adapters if specified
        if self.config.summarizer_lora_path:
            logger.info(f"Loading LoRA adapters from: {self.config.summarizer_lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.summarizer_lora_path)
            logger.info("LoRA adapters loaded successfully")
        
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
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs["input_ids"].shape[1]
        
        # Generate summary
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.7,  # non-thinking mode
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (after the input tokens)
        generated_tokens = generated_ids[0][input_length:]
        summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
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


class InferenceSystem:
    """Main system that coordinates reasoner, retriever, and summarizer."""
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.reasoner = Reasoner(config.reasoner_model_name, config)
        self.summarizer = Summarizer(config.summarizer_model_name, config)
        self.retriever = self._load_retriever()
        
        # Load probe if enabled
        self.probe = None
        self.pca = None
        self.scaler = None
        if self.config.use_probe:
            self._load_probe()
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _load_retriever(self):
        if self.config.retriever_type == "bm25":
            return BM25Retriever(self.config.retriever_index_path, self.config.top_k_docs)
        elif self.config.retriever_type == "e5":
            return E5Retriever(self.config.retriever_index_path, self.config.e5_model_path)
        else:
            raise ValueError(f"Unsupported retriever type: {self.config.retriever_type}")
    
    def _load_probe(self):
        """Load the trained knowledge probe and associated components."""
        import pickle
        
        logger.info(f"Loading probe from {self.config.probe_path}")
        
        try:
            # Load probe configuration
            config_path = os.path.join(self.config.probe_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Probe config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                probe_config = json.load(f)
            
            # Extract probe configuration
            self.probe_layer = probe_config.get("probe_layer", 32)
            self.probe_input_dim = probe_config.get("input_dim", 5120)
            self.probe_pca_dim = probe_config.get("pca_dim", 64)
            
            logger.info(f"Probe config loaded: layer={self.probe_layer}, input_dim={self.probe_input_dim}, pca_dim={self.probe_pca_dim}")
            
            # Load probe model
            probe_model_path = os.path.join(self.config.probe_path, "best_probe.pth")
            if not os.path.exists(probe_model_path):
                probe_model_path = os.path.join(self.config.probe_path, "final_probe.pth")
            
            if not os.path.exists(probe_model_path):
                raise FileNotFoundError(f"No probe model found in {self.config.probe_path}")
            
            # Load PCA and scaler
            pca_path = os.path.join(self.config.probe_path, "pca.pkl")
            scaler_path = os.path.join(self.config.probe_path, "scaler.pkl")
            
            if not os.path.exists(pca_path) or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"PCA or scaler files not found in {self.config.probe_path}")
            
            # Load components
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load probe model
            from train_probe import KnowledgeProbe
            self.probe = KnowledgeProbe(input_dim=self.pca.n_components_)
            self.probe.load_state_dict(torch.load(probe_model_path, map_location='cpu'))
            self.probe.eval()
            
            logger.info(f"Probe loaded successfully with {self.pca.n_components_} PCA components")
            
        except Exception as e:
            logger.error(f"Failed to load probe: {e}")
            raise
    
    def _get_probe_confidence(self, question: str) -> float:
        """
        Get confidence score from the probe for a given question.
        
        Args:
            question: The question to evaluate
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not self.probe:
            return 0.0
        
        try:
            # Extract hidden states from the reasoner model
            input_text = f"<search> {question} </search>"
            inputs = self.reasoner.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.reasoner.model.device)
            
            with torch.no_grad():
                outputs = self.reasoner.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # Check if target layer is within range
                if self.probe_layer >= len(hidden_states):
                    raise ValueError(f"Target layer {self.probe_layer} is out of range. Model has {len(hidden_states)} layers.")
                
                # Extract hidden state at the target layer and last token position
                last_token_pos = input_ids.shape[1] - 1
                hidden_state = hidden_states[self.probe_layer][0, last_token_pos, :].float().cpu().numpy()
            
            # Apply PCA and scaling
            hidden_state_scaled = self.scaler.transform(hidden_state.reshape(1, -1))
            hidden_state_pca = self.pca.transform(hidden_state_scaled)
            
            # Get probe prediction
            with torch.no_grad():
                probe_input = torch.FloatTensor(hidden_state_pca)
                confidence = self.probe(probe_input).item()
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Failed to get probe confidence for question: {question[:50]}... Error: {e}")
            return 0.0
    
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
    
    def inference_one_turn(self, active_questions: List[Dict[str, Any]], turn_pbar: Optional[tqdm] = None) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Perform one turn of inference for all active questions.
        
        Args:
            active_questions: List of question data with current sequences
            turn_pbar: Optional progress bar for turn-level progress
            
        Returns:
            Tuple of (updated_active_questions, search_queries_dict)
        """
        search_queries = {}  # question_id -> search_query
        
        # Create progress bar for processing individual questions in this turn
        question_pbar = tqdm(active_questions, desc="Processing questions", unit="question", position=2, leave=False)
        
        for question_data in question_pbar:
            # Update progress bar to show current question
            question_pbar.set_description(f"Processing question {question_data['id']}")
            
            # Generate response using the current sequence
            response = self.reasoner.generate_response(question_data["sequence"])
            
            # Extract search query or answer
            search_query = self._extract_search_query(response)
            answer = self._extract_answer(response)
            
            # Record turn info
            turn_info = {
                "turn": len(question_data["turns"]) + 1,
                "response": response,
                "search_query": search_query,
                "answer": answer
            }
            question_data["turns"].append(turn_info)
            
            # Update sequence with the current response
            question_data["sequence"] += response
            question_data["response"] += response
            
            if search_query:
                # Need to search, collect search query
                search_queries[question_data["id"]] = search_query
        
        question_pbar.close()
        return active_questions, search_queries
    
    def process_retrievals(self, active_questions: List[Dict[str, Any]], search_queries: Dict[str, str], probe_counters: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Process all search queries and update active questions with retrieved information.
        Includes probe-based confidence handling if probe is enabled.
        
        Args:
            active_questions: List of active question data
            search_queries: Dictionary mapping question_id to search query
            probe_counters: Optional probe statistics counter (for compatibility with process_dataset)
            
        Returns:
            Updated active_questions with retrieval information
        """
        if not search_queries:
            return active_questions
        
        # Collect all unique search queries to avoid duplicate retrievals
        unique_queries = list(set(search_queries.values()))
        query_to_questions = {}
        
        for question_id, query in search_queries.items():
            if query not in query_to_questions:
                query_to_questions[query] = []
            query_to_questions[query].append(question_id)
        
        # Create progress bar for processing search queries
        query_pbar = tqdm(unique_queries, desc="Processing search queries", unit="query", position=3, leave=False)
        
        # Process each unique query
        for query in query_pbar:
            try:
                # Update progress bar to show current query
                query_pbar.set_description(f"Processing query: {query[:30]}...")
                
                # Check probe confidence if probe is enabled
                should_retrieve = True
                probe_confidence = 0.0
                
                if self.config.use_probe:
                    # Get the original question for this search query
                    original_question = None
                    for question_data in active_questions:
                        if question_data["id"] in query_to_questions[query]:
                            original_question = question_data["question"]
                            break
                    
                    if original_question:
                        probe_confidence = self._get_probe_confidence(original_question)
                        should_retrieve = probe_confidence < self.config.probe_confidence_threshold
                        
                        logger.info(f"Query: {query[:50]}... | Probe confidence: {probe_confidence:.4f} | Threshold: {self.config.probe_confidence_threshold} | Should retrieve: {should_retrieve}")
                
                if should_retrieve:
                    # Count performed retrieval
                    if probe_counters is not None:
                        probe_counters["performed_retrievals"] += 1
                    
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
                                
                                # Update turn info with retrieval results
                                for turn_info in question_data["turns"]:
                                    if turn_info["search_query"] == query:
                                        turn_info["retrieved_docs"] = [str(doc) for doc in retrieved_docs]
                                        turn_info["summary"] = summary
                                        turn_info["probe_confidence"] = probe_confidence
                                        turn_info["retrieval_skipped"] = False
                                        break
                                break
                else:
                    # Count skipped retrieval
                    if probe_counters is not None:
                        probe_counters["skipped_retrievals"] += 1
                    
                    # High confidence - let model answer directly in next turn
                    # Add probe confidence info and mark for direct answer generation
                    for question_id in query_to_questions[query]:
                        for question_data in active_questions:
                            if question_data["id"] == question_id:
                                for turn_info in question_data["turns"]:
                                    if turn_info["search_query"] == query:
                                        turn_info["probe_confidence"] = probe_confidence
                                        turn_info["retrieval_skipped"] = True
                                        turn_info["retrieved_docs"] = []
                                        turn_info["summary"] = f"High probe confidence ({probe_confidence:.4f} >= {self.config.probe_confidence_threshold}) - model will answer directly"
                                        break
                                
                                # Add a prompt to encourage the model to answer directly
                                # This simulates the model having "retrieved" information from its internal knowledge
                                internal_knowledge_prompt = f"Based on my internal knowledge (confidence: {probe_confidence:.4f}), I can answer this question directly without external documents. I will put the answer in <information> </information>. <information>"
                                question_data["sequence"] += internal_knowledge_prompt
                                question_data["response"] += internal_knowledge_prompt
                                break
                    
                    logger.info(f"High confidence ({probe_confidence:.4f} >= {self.config.probe_confidence_threshold}) - model will answer directly for query '{query[:50]}...'")
                    
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        query_pbar.close()
        
        # Update probe statistics for answered questions (if probe_counters provided)
        if probe_counters is not None and self.config.use_probe:
            for question_data in active_questions:
                if question_data.get("answer"):
                    # Check if this question had any skipped retrievals
                    had_skipped_retrieval = any(
                        turn.get("retrieval_skipped", False) 
                        for turn in question_data["turns"] 
                        if turn.get("search_query")
                    )
                    
                    if had_skipped_retrieval:
                        probe_counters["self_answered"] += 1
                    else:
                        # Check if it had any performed retrievals
                        had_performed_retrieval = any(
                            not turn.get("retrieval_skipped", True) 
                            for turn in question_data["turns"] 
                            if turn.get("search_query")
                        )
                        if had_performed_retrieval:
                            probe_counters["retrieved_answered"] += 1
        
        return active_questions
    
    def inference(self, questions: List[Dict[str, Any]], max_turns: Optional[int] = None, probe_counters: Optional[Dict[str, int]] = None, progress_bar: Optional[tqdm] = None) -> List[Dict[str, Any]]:
        """
        Perform full inference for a list of questions.
        
        Args:
            questions: List of question data with 'id', 'question', 'golden_answers'
            max_turns: Maximum number of turns (uses config default if None)
            probe_counters: Optional probe statistics counter (for compatibility with process_dataset)
            progress_bar: Optional tqdm progress bar for tracking question completion
            
        Returns:
            List of completed question results
        """
        if max_turns is None:
            max_turns = self.config.max_turns
        
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
                "sequence": initial_sequence,
                "response": "",
                "turns": [],
                "final_turn": 0,
                "answer": None,
                "error": None
            })
        
        max_turn_warning = "Time is up. I am not allowed to search anymore. I should give a final answer now with the information I have."
        completed_questions = []
        
        # Process questions in turns with progress tracking
        turn_pbar = tqdm(range(max_turns + 1), desc="Inference turns", unit="turn", position=1, leave=False) if progress_bar else range(max_turns + 1)
        for turn_num in turn_pbar:
            if not active_questions:
                break
            
            # Update turn progress bar description
            if progress_bar and hasattr(turn_pbar, 'set_description'):
                turn_pbar.set_description(f"Turn {turn_num + 1}/{max_turns + 1} (Active: {len(active_questions)})")
            
            # Step 1: Generate responses for all active questions
            active_questions, search_queries = self.inference_one_turn(active_questions, turn_pbar)
            
            # Update probe counters for search queries
            if probe_counters is not None and search_queries:
                probe_counters["total_search_queries"] += len(search_queries)
            
            # Step 2: Check for completed questions (answered or error)
            new_completed = []
            for question_data in active_questions:
                last_turn = question_data["turns"][-1] if question_data["turns"] else {}
                answer = last_turn.get("answer")
                search_query = last_turn.get("search_query")
                
                if answer:
                    # Question answered
                    question_data["answer"] = answer
                    question_data["final_turn"] = turn_num + 1
                    logger.info(f"Question {question_data['id']} completed in turn {turn_num + 1}")
                    new_completed.append(question_data)
                    # Update progress bar with question ID
                    if progress_bar:
                        progress_bar.set_description(f"Completed question {question_data['id']}")
                        progress_bar.update(1)
                elif not search_query:
                    # No search query or answer found
                    question_data["error"] = "No search query or answer found"
                    question_data["final_turn"] = turn_num + 1
                    logger.info(f"Question {question_data['id']} completed with error: No search query or answer found")
                    new_completed.append(question_data)
                    # Update progress bar with question ID
                    if progress_bar:
                        progress_bar.set_description(f"Completed question {question_data['id']} (error)")
                        progress_bar.update(1)
            
            # Add completed questions to results
            completed_questions.extend(new_completed)
            
            # Remove completed questions from active list
            active_questions = [q for q in active_questions if q not in new_completed]
            
            # Step 3: Process retrievals if any search queries
            if search_queries:
                active_questions = self.process_retrievals(active_questions, search_queries, probe_counters)
            
            # Step 4: Check if max turns reached
            if turn_num == max_turns - 1 and active_questions:
                for question_data in active_questions:
                    question_data["sequence"] += max_turn_warning
            elif turn_num == max_turns:
                for question_data in active_questions:
                    question_data["error"] = "Max turns reached"
                    question_data["final_turn"] = max_turns
                    logger.info(f"Question {question_data['id']} completed with error: Max turns reached")
                    completed_questions.append(question_data)
                    # Update progress bar with question ID
                    if progress_bar:
                        progress_bar.set_description(f"Completed question {question_data['id']} (max turns)")
                        progress_bar.update(1)
                active_questions = []
        
        # Calculate metrics for all results
        for result in completed_questions:
            if result["answer"]:
                from utils import calculate_metrics
                metrics = calculate_metrics(result["answer"], result["golden_answers"])
                result["metrics"] = metrics
            else:
                result["metrics"] = {"em": 0.0, "f1": 0.0, "cover_match": 0.0}
        
        # Log probe statistics if probe was used
        if self.config.use_probe and probe_counters is not None:
            logger.info("="*60)
            logger.info("PROBE STATISTICS SUMMARY")
            logger.info("="*60)
            logger.info(f"Total Search Queries: {probe_counters['total_search_queries']}")
            logger.info(f"Skipped Retrievals: {probe_counters['skipped_retrievals']}")
            logger.info(f"Performed Retrievals: {probe_counters['performed_retrievals']}")
            logger.info(f"Self-Answered: {probe_counters['self_answered']}")
            logger.info(f"Retrieved-Answered: {probe_counters['retrieved_answered']}")
            logger.info(f"Confidence Threshold: {self.config.probe_confidence_threshold}")
            logger.info("="*60)
        
        return completed_questions
    

    
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
                elif self.config.dataset_name == "2wikimultihop":  # 2wikimultihop
                    questions.append({
                        "id": data["_id"],
                        "question": data["question"],
                        "golden_answers": data["answer"]
                    })
        
        logger.info(f"Loaded {len(questions)} questions from dataset")
        
        # Limit samples if specified
        if self.config.start_sample is not None or self.config.end_sample is not None:
            start = (self.config.start_sample) if self.config.start_sample is not None else 0
            end = self.config.end_sample if self.config.end_sample is not None else len(questions)
            original_count = len(questions)
            questions = questions[start:end]
            logger.info(f"Selected questions from {start+1} to {end} (total: {len(questions)}, original: {original_count})")
        elif self.config.max_samples:
            original_count = len(questions)
            questions = questions[:self.config.max_samples]
            logger.info(f"Limited to {len(questions)} questions (max_samples: {self.config.max_samples}, original: {original_count})")
        
        logger.info(f"Processing {len(questions)} questions")
        
        # Initialize probe statistics counters
        probe_counters = {
            "total_search_queries": 0,
            "skipped_retrievals": 0,
            "performed_retrievals": 0,
            "self_answered": 0,
            "retrieved_answered": 0
        }
        
        # Use the modular inference method with progress tracking
        with tqdm(total=len(questions), desc="Processing questions", unit="question") as pbar:
            all_results = self.inference(questions, probe_counters=probe_counters, progress_bar=pbar)
        
        # Add probe statistics to all results for later analysis (preserve original behavior)
        if self.config.use_probe:
            for result in all_results:
                result["probe_counters"] = probe_counters
        
        # Save intermediate results if requested
        if self.config.save_intermediate:
            self._save_intermediate_results(all_results)
        
        logger.info(f"Processed {len(all_results)} questions successfully")
        return all_results
    
    def _save_intermediate_results(self, results: List[Dict[str, Any]]):
        """Save intermediate results to file."""
        # Sort results by question ID/index for consistent output order
        sorted_results = sorted(results, key=lambda x: x.get("id", ""))
        
        output_file = os.path.join(self.config.output_dir, "intermediate_results.json")
        with open(output_file, 'w') as f:
            json.dump(sorted_results, f, indent=2)
    
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
            
            del clean_result["sequence"]
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
        avg_cover_match = sum(r["metrics"]["cover_match"] for r in clean_results) / total_questions
        
        avg_turns = sum(r["final_turn"] for r in clean_results) / total_questions
        
        summary = {
            "dataset": dataset_name,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "answer_rate": answered_questions / total_questions,
            "average_em": avg_em,
            "average_f1": avg_f1,
            "average_cover_match": avg_cover_match,
            "average_turns": avg_turns,
            "config": self.config.__dict__
        }
        
        summary_file = os.path.join(self.config.output_dir, f"{dataset_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"Average EM: {avg_em:.3f}, Average F1: {avg_f1:.3f}, Average Cover Match: {avg_cover_match:.3f}")


def main():
    """Main function to run Search-o1 system."""
    # Start timing
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Agentic RAG Inference System")
    
    # Model settings
    parser.add_argument("--reasoner-model", default="Qwen/Qwen3-32B", help="Reasoner model name")
    parser.add_argument("--summarizer-model", default="Qwen/Qwen3-32B", help="Summarizer model name")
    parser.add_argument("--reasoner-lora-path", default=None, help="Path to LoRA adapters for reasoner")
    parser.add_argument("--summarizer-lora-path", default=None, help="Path to LoRA adapters for summarizer")
    parser.add_argument("--retriever-type", default="bm25", choices=["bm25", "e5"], help="Retriever type")
    parser.add_argument("--retriever-index-path", default="indexes/bm25", help="Path to retriever index")
    parser.add_argument("--e5-model-path", default="intfloat/e5-large-v2", help="Path to E5 model for retrieval")
    
    # Generation settings
    parser.add_argument("--max-turns", type=int, default=5, help="Maximum number of turns")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--greedy-thinking", action="store_true", help="Use greedy decoding in thinking mode (no sampling)")
    
    # Retrieval settings
    parser.add_argument("--top-k-docs", type=int, default=10, help="Number of documents to retrieve")
    
    # Dataset settings
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "2wikimultihop"], help="Dataset to use")
    parser.add_argument("--split", default="dev", choices=["train", "dev"], help="Dataset split to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--start-sample", type=int, default=None, help="Start index (1-based, inclusive) of samples to process")
    parser.add_argument("--end-sample", type=int, default=None, help="End index (1-based, inclusive) of samples to process")
    
    # Output settings
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-generated if not specified)")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    
    # Probe settings
    parser.add_argument("--use-probe", action="store_true", help="Enable probe-based inference mode")
    parser.add_argument("--probe-path", default="probe/", help="Path to probe directory containing trained model")
    parser.add_argument("--probe-confidence-threshold", type=float, default=0.7, help="Confidence threshold above which retrieval is skipped")
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        if args.use_probe:
            args.output_dir = f"output/search_o1_probe_{args.probe_confidence_threshold}"
        else:
            args.output_dir = "output/search_o1"
    
    # Create config
    config = InferenceConfig(
        reasoner_model_name=args.reasoner_model,
        summarizer_model_name=args.summarizer_model,
        reasoner_lora_path=args.reasoner_lora_path,
        summarizer_lora_path=args.summarizer_lora_path,
        retriever_type=args.retriever_type,
        retriever_index_path=args.retriever_index_path,
        e5_model_path=args.e5_model_path,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        greedy_thinking=args.greedy_thinking,
        top_k_docs=args.top_k_docs,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate,
        start_sample=args.start_sample,
        end_sample=args.end_sample,
        use_probe=args.use_probe,
        probe_path=args.probe_path,
        probe_confidence_threshold=args.probe_confidence_threshold
    )
    
    # Create system
    system = InferenceSystem(config)
    
    # Log LoRA configuration if enabled
    if config.reasoner_lora_path:
        logger.info(f"LoRA-enabled reasoner: {config.reasoner_lora_path}")
    else:
        logger.info("Standard reasoner (no LoRA)")
    
    if config.summarizer_lora_path:
        logger.info(f"LoRA-enabled summarizer: {config.summarizer_lora_path}")
    else:
        logger.info("Standard summarizer (no LoRA)")
    
    # Log probe configuration if enabled
    if config.use_probe:
        logger.info(f"Probe-based inference mode enabled")
        logger.info(f"Probe path: {config.probe_path}")
        logger.info(f"Confidence threshold: {config.probe_confidence_threshold}")
        logger.info(f"Questions with confidence >= {config.probe_confidence_threshold} will skip retrieval")
        logger.info("Probe configuration (layer, PCA dim, etc.) will be loaded from probe config file")
    else:
        logger.info("Standard inference mode (no probe)")
    
    # Log generation configuration
    if config.greedy_thinking:
        logger.info("Greedy decoding enabled for thinking mode (deterministic reasoning)")
    else:
        logger.info("Sampling-based decoding enabled for thinking mode (creative reasoning)")
    
    # Determine dataset path
    dataset_path = f"data/{args.dataset}/{args.split}.jsonl"
    
    # Process dataset
    results = system.process_dataset(dataset_path)
    
    # Sort results by question ID/index for consistent output order
    results.sort(key=lambda x: x.get("id", ""))
    
    # Save results
    system.save_final_results(results, args.dataset)
    
    # Calculate and log total running time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert to hours, minutes, seconds for readability
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    logger.info("="*60)
    logger.info("TIMING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Running Time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.2f} seconds)")
    logger.info(f"Average Time per Question: {total_time/len(results):.2f} seconds")
    logger.info("="*60)
    
    logger.info("Search-o1 processing completed!")


if __name__ == "__main__":
    main()