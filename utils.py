import yaml
import re
import string
import collections
from typing import Optional, Tuple, Dict, Any
import hydra
from hydra.utils import to_absolute_path

TEMPLATE_CACHE: Dict[str, str] = {}

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
    
    # Use regex to find <search> or <answer> tags and extract content
    pattern = r"(.*?)<(search|answer)>(.*?)</\2>"
    match = re.search(pattern, response, re.DOTALL)
    
    if not match:
        return response, None, None
    
    thought = match.group(1).strip()
    content = match.group(3).strip()
    action = match.group(2)
    
    if action == 'search':
        return thought, content, None
    elif action == 'answer':
        return thought, None, content
    else:
        raise ValueError(f"Invalid action type: {action}")

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

def normalize_answer_qa(text: str) -> str:
    """
    Normalize text for QA evaluation by removing articles, fixing whitespace,
    converting to lowercase, and removing punctuation.
    
    Args:
        text: Input text to normalize.
        
    Returns:
        Normalized text string.
    """
    if not text:
        return ""
    
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', '', text)
    
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    
    def lower(text: str) -> str:
        return text.lower()
    
    def remove_punc(text: str) -> str:
        return re.sub(f"[{string.punctuation}]", "", text)
    
    return remove_articles(white_space_fix(remove_punc(lower(text))))

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
        reasoning_match = re.search(r'(.*?)</think>', response, re.DOTALL)
        reasoning_output = reasoning_match.group(1).strip() if reasoning_match else response
        answer_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        answer_output = answer_match.group(1).strip() if answer_match else ""
        
    else:
        raise ValueError(f"Unsupported LLM path: {llm_path}")
    
    return reasoning_output, answer_output