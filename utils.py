import yaml
import re
import string
import collections
import hydra
from hydra.utils import to_absolute_path

TEMPLATE_CACHE = {}

def load_user_message_template(template_dir, template_name):
    # find the template file
    key = f"{template_dir}/{template_name}"

    #if the template file is found, return the template directly from the cache
    if key in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[key]
    
    # else if the template file is not found, load the template form the file to the cache
    prompt_path = to_absolute_path(f"{template_dir}/{template_name}.yaml")
    with open(prompt_path, 'r') as f:
        prompt = yaml.load(f, Loader=yaml.FullLoader)['user_prompt']
    TEMPLATE_CACHE[key] = prompt
    return prompt

def load_sys_message(llm_path):
    # load the system message from the LLM path
    if llm_path is None:
        return ""
    elif 'deepseek-r1' in llm_path.lower():
        return ""
    elif "llama" in llm_path.lower():
        return ""
    elif "qwen" in llm_path.lower() or "qwq" in llm_path.lower():
        return "You are a helpful and harmless assistant, You are Quen developed bu Alibaba, You shold think step by step and provide a detailed answer."
    
def parse_reasoning_generation(response):
    # check if the response is in a string format(cleared the format)
    if isinstance (response, str):

        # use regex to find the <search> or <answer> tags and extract the content between them
        pattern = r"(.*?)<(search|answer)>(.*?)</\2>" # hardcode for now, maybe back to modify late
        match = re.search(pattern, response, re.DOTALL)

        # if no tags are found, return the original response
        if not match:
            return response, None, None

        # if any the tags are found
        thought = match.group(1).strip()
        content = match.group(3).strip()
        action = match.group(2)
        
        # determine the action type the the downstream task, return the thought and the content of the action. The order is thought, search content, answer content
        if action == 'search':
            return thought, content, None
        elif action == 'answer':
            return thought, None, content
        else:
            raise ValueError(f"Invalid action type: {type(response)}")
        
def parse_summary_generation(response):
    # define the Extracted information tag for the summary generation
    tag = "### Extracted Information"

    # check if the response is in a string format
    if isinstance(response, str):

        # check if the tag is in the response
        if tag in response:
            # if the tag is in the response, return the information after the tag
            information = response.split(tag)[1].strip()
            return response, information
        else:
            # Fallback: try to extract meaningful content even without the tag
            response_clean = response.strip()
            if len(response_clean) > 0:
                # If there's content but no tag, return it with a warning
                return response, f"[NO TAG FOUND] {response_clean[:200]}..."
            else:
                return response, "No useful information is extracted"
    else:
        raise ValueError(f"Invalid response type: {type(response)}")
    
"""
Evaluation metrics utilizaitons
"""
    
def string_f1(pred, target):
    # Tokenize the predicted respose and the targeted response
    pred_tokens = pred.split()
    target_tokens = target.split()

    # count the tokens that are in both the predicted and the targeted response and get the cross set which the tokens are shown in both and have the same frequency
    common = collections.Counter(pred_tokens) & collections.Counter(target_tokens)

    # count the number of tokens that are in the common set
    num_same = sum(common.values())

    # if no tokens are in both the predicted and the targeted response, return 0
    if num_same == 0:
        return 0
    
    # calculate the precision and recall
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(target_tokens)

    # calculate the f1 score
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer_qa(text):
    # define a function to remove the articles from the text
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', '', text)
    
    # define a function to fix the white space
    def white_space_fix(text):
        return ' '.join(text.split())
    
    # define a function to lowercase the text
    def lower(text):
        return text.lower()
    
    # define a function to remove the punctuation from the text
    def remove_punc(text):
        return re.sub(f"[{string.punctuation}]", "", text)
    
    # return the normalized text
    return remove_articles(white_space_fix(remove_punc(lower(text))))

def extract_reasoning_and_answer(response, llm_path):
    # handle the deepseek-r1 model specific response format
    if "deepseek-r1" in llm_path.lower():
        reasoning_match = re.search(r'(.*?)</think>', response, re.DOTALL)
        reasoning_output = reasoning_match.group(1).strip() if reasoning_match else response
        anwer_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        answer_output = anwer_match.group(1).strip() if anwer_match else ""

    elif "qwen" in llm_path.lower() or "qwq" in llm_path.lower():
        reasoning_match = re.search(r'(.*?)</think>', response, re.DOTALL)
        reasoning_output = reasoning_match.group(1).strip() if reasoning_match else response
        anwer_match = re.search(r'Final Answer:\s*(.*)', response, re.DOTALL)
        answer_output = anwer_match.group(1).strip() if anwer_match else ""
        
    else:
        raise ValueError(f"Invalid LLM path: {llm_path}")
    return reasoning_output, answer_output