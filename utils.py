import yaml
import re
import string
import collections
import json
from typing import Optional, Tuple, Dict, Any, List
import hydra
from hydra.utils import to_absolute_path

TEMPLATE_CACHE: Dict[str, str] = {}

def load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def load_dataset(data_path: str, max_data_samples: Optional[int] = None) -> List[dict]:
    items = load_json_or_jsonl(data_path)
    if not items:
        raise ValueError(f"No items loaded from {data_path}")
    # Limit data samples if specified
    if max_data_samples is not None and max_data_samples > 0:
        items = items[:max_data_samples]
    return items

def load_user_message_template(template_dir: str, template_name: str) -> str:
    """
    Load a user message template from a YAML file with caching.
    
    Args:
        template_dir: Directory containing template files.
        template_name: Name of the template file (without .yaml extension).
        
    Returns:
        The user prompt template as a string.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
        KeyError: If the YAML file doesn't contain 'user_prompt' key.
    """
    key = f"{template_dir}/{template_name}"
    
    if key in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[key]
    
    prompt_path = to_absolute_path(f"{template_dir}/{template_name}.yaml")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = yaml.safe_load(f)['user_prompt']
        TEMPLATE_CACHE[key] = prompt
        return prompt
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {prompt_path}")
    except KeyError:
        raise KeyError(f"Template file {prompt_path} does not contain 'user_prompt' key")

def load_sys_message(llm_path: Optional[str]) -> str:
    """
    Load system message based on the LLM model type.
    
    Args:
        llm_path: Path or name of the LLM model.
        
    Returns:
        System message string appropriate for the model.
    """
    if llm_path is None:
        return ""
    
    llm_path_lower = llm_path.lower()
    
    if 'deepseek-r1' in llm_path_lower:
        return ""
    elif "llama" in llm_path_lower:
        return ""
    elif "qwen" in llm_path_lower or "qwq" in llm_path_lower:
        return ("You are a helpful and harmless assistant. You are Qwen developed by Alibaba. "
                "You should think step by step and provide a detailed answer.")
    else:
        return ""

def parse_reasoning_generation(response: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse reasoning generation output to extract thought, search query, and answer.
    
    Args:
        response: The generated response string.
        
    Returns:
        Tuple of (thought, search_query, answer) where search_query and answer may be None.
        
    Raises:
        ValueError: If response is not a string or contains invalid action type.
    """
    if not isinstance(response, str):
        raise ValueError(f"Response must be a string, got {type(response)}")
    
    # Find the last <answer> tag to get the final answer
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = list(re.finditer(answer_pattern, response, re.DOTALL))
    
    if answer_matches:
        # Get the last answer
        last_answer_match = answer_matches[-1]
        answer = last_answer_match.group(1).strip()
        
        # Get the thought part before the last answer
        thought_end = last_answer_match.start()
        thought = response[:thought_end].strip()
        
        return thought, None, answer
    
    # If no answer tag found, look for search tags
    search_pattern = r"<search>(.*?)</search>"
    search_matches = list(re.finditer(search_pattern, response, re.DOTALL))
    
    if search_matches:
        # Get the last search
        last_search_match = search_matches[-1]
        search_query = last_search_match.group(1).strip()
        
        # Get the thought part before the last search
        thought_end = last_search_match.start()
        thought = response[:thought_end].strip()
        
        return thought, search_query, None
    
    # If no tags found, return the whole response as thought
    return response, None, None

def parse_summary_generation(response: str) -> Tuple[str, str]:
    """
    Parse summary generation output to extract extracted information.
    
    Args:
        response: The generated summary response string.
        
    Returns:
        Tuple of (original_response, extracted_information).
        
    Raises:
        ValueError: If response is not a string.
    """
    if not isinstance(response, str):
        raise ValueError(f"Response must be a string, got {type(response)}")
    
    tag = "### Extracted Information"
    
    if tag in response:
        # Extract information after the tag
        parts = response.split(tag, 1)
        if len(parts) > 1:
            information = parts[1].strip()
        else:
            information = ""
        return response, information
    else:
        return response, "No useful information is extracted"

def extract_answer(text: str) -> str:
    """Extract content inside <answer>...</answer>. If missing, fallback to last non-empty line."""
    ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
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


def is_exact_match(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)

"""
Evaluation metrics utilizaitons
"""
    
def string_f1(pred: str, target: str) -> float:
    """
    Calculate string F1 score between prediction and target.
    
    Args:
        pred: Predicted text.
        target: Target text.
        
    Returns:
        F1 score as a float between 0 and 1.
    """
    if not pred or not target:
        return 0.0
    
    pred = normalize_text(pred)
    target = normalize_text(target)
    
    pred_tokens = pred.split()
    target_tokens = target.split()
    
    if not pred_tokens or not target_tokens:
        return 0.0
    
    # Count common tokens with same frequency
    common = collections.Counter(pred_tokens) & collections.Counter(target_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    # Calculate precision and recall
    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def cover_match(pred: str, target: str) -> float:
    """
    Calculate cover match (token overlap ratio) between prediction and target.
    Measures the proportion of prediction tokens that are covered by the ground-truth answer tokens.
    Args:
        pred: Predicted text.
        target: Ground-truth text.
    Returns:
        Cover match score as a float between 0 and 1.
    """
    pred = normalize_text(pred)
    target = normalize_text(target)
    
    if not pred or not target:
        return 0.0
    pred_tokens = pred.split()
    target_tokens = target.split()
    if not pred_tokens:
        return 0.0
    # Count common tokens with same frequency
    common = collections.Counter(pred_tokens) & collections.Counter(target_tokens)
    num_same = sum(common.values())
    return num_same / len(pred_tokens) if pred_tokens else 0.0

def extract_reasoning_and_answer(response: str, llm_path: str) -> Tuple[str, str]:
    """
    Extract reasoning and answer from model-specific response formats.
    
    Args:
        response: The model response string.
        llm_path: Path or name of the LLM model.
        
    Returns:
        Tuple of (reasoning_output, answer_output).
        
    Raises:
        ValueError: If LLM path is not supported.
    """
    if not isinstance(response, str):
        raise ValueError(f"Response must be a string, got {type(response)}")
    
    llm_path_lower = llm_path.lower()
    
    if "deepseek-r1" in llm_path_lower:
        reasoning_match = re.search(r'(.*?)</think>', response, re.DOTALL)
        reasoning_output = reasoning_match.group(1).strip() if reasoning_match else response
        answer_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        answer_output = answer_match.group(1).strip() if answer_match else ""
        
    elif "qwen" in llm_path_lower or "qwq" in llm_path_lower:
        # Extract reasoning (everything before the last <answer> tag)
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_matches = list(re.finditer(answer_pattern, response, re.DOTALL))
        
        if answer_matches:
            # Get the last answer
            last_answer_match = answer_matches[-1]
            answer_output = last_answer_match.group(1).strip()
            
            # Get the reasoning part before the last answer
            reasoning_end = last_answer_match.start()
            reasoning_output = response[:reasoning_end].strip()
        else:
            # Fallback to old format
            reasoning_match = re.search(r'(.*?)</think>', response, re.DOTALL)
            reasoning_output = reasoning_match.group(1).strip() if reasoning_match else response
            answer_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
            answer_output = answer_match.group(1).strip() if answer_match else ""
        
    else:
        raise ValueError(f"Unsupported LLM path: {llm_path}")
    
    return reasoning_output, answer_output


def calculate_metrics(prediction: str, golden_answers: list) -> Dict[str, float]:
    """
    Calculate evaluation metrics (EM, F1, and cover_match) for a prediction against golden answers.
    Args:
        prediction: The predicted answer string.
        golden_answers: List of golden answer strings.
    Returns:
        Dictionary containing 'em', 'f1', and 'cover_match' scores.
    """
    if not prediction:
        return {"em": 0.0, "f1": 0.0, "cover_match": 0.0}
    # Normalize prediction
    pred_normalized = normalize_text(prediction)
    # Calculate metrics against each golden answer
    best_em = 0.0
    best_f1 = 0.0
    best_cover = 0.0
    for golden_answer in golden_answers:
        if not golden_answer:
            continue
        # Normalize golden answer
        gold_normalized = normalize_text(golden_answer)
        # Calculate exact match
        em_score = 1.0 if pred_normalized == gold_normalized else 0.0
        # Calculate F1 score
        f1_score = string_f1(pred_normalized, gold_normalized)
        # Calculate cover match
        cover_score = cover_match(pred_normalized, gold_normalized)
        # Update best scores
        best_em = max(best_em, em_score)
        best_f1 = max(best_f1, f1_score)
        best_cover = max(best_cover, cover_score)
    return {"em": best_em, "f1": best_f1, "cover_match": best_cover}


