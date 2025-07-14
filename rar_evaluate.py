import re
from typing import List, Dict, Tuple, Any
from utils import normalize_answer_qa, string_f1

def run_evaluation(
    data: List[Dict[str, Any]],
    all_sequences: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Evaluate predictions against gold answers with multiple metrics.
    Args:
        data: List of data items, each with 'question' and 'golden_answers'.
        all_sequences: List of prediction dicts, each with 'output', 'answer', 'finished'.
    Returns:
        metrics_list: List of per-example metric dicts.
        metrics_agg: Dict of averaged metrics over the dataset.
    """
    metrics_list: List[Dict[str, float]] = []
    metrics_agg: Dict[str, float] = {
        "em": 0.0,
        "cover_match": 0.0,
        "str_f1": 0.0,
        "valid": 0.0,
        "finished": 0.0,
        "num_retrieval": 0.0,
        "num_fail_retrieval": 0.0,
    }

    for item, seq in zip(data, all_sequences):
        output = seq.get("output", "")
        pred = seq.get('answer', "")
        gold_answers = item.get("golden_answers", [])

        if not pred:
            em = cover_match = str_f1 = valid = 0.0
        else:
            norm_pred = normalize_answer_qa(pred)
            norm_gold = [normalize_answer_qa(g) for g in gold_answers]
            em = float(norm_pred in norm_gold)
            cover_match = float(any(g in norm_pred for g in norm_gold))
            str_f1 = max([string_f1(norm_pred, g) for g in norm_gold]) if norm_gold else 0.0
            valid = 1.0

        num_search = float(len(re.findall(r"<search>.*?</search>", output, flags=re.DOTALL)))
        num_fail = float(len(re.findall(r"<information>\s*No helpful information found\s*</information>", output, flags=re.DOTALL)))

        metrics_agg["num_retrieval"] += num_search
        metrics_agg["num_fail_retrieval"] += num_fail
        metrics_agg["em"] += em
        metrics_agg["cover_match"] += cover_match
        metrics_agg["str_f1"] += str_f1
        metrics_agg["valid"] += valid
        metrics_agg["finished"] += float(seq.get('finished', 0))

        metrics_list.append({
            "em": em,
            "cover_match": cover_match,
            "str_f1": str_f1,
            "valid": valid,
            "finished": float(seq.get('finished', 0)),
            "num_retrieval": num_search,
            "num_fail_retrieval": num_fail,
        })

    n = float(len(data)) if data else 1.0
    for k in metrics_agg:
        metrics_agg[k] /= n

    return metrics_list, metrics_agg

def run_evaluation_simple(
    data: List[Dict[str, Any]],
    all_sequences: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Simpler evaluation: only EM, cover_match, and string F1.
    Args:
        data: List of data items, each with 'question' and 'golden_answers'.
        all_sequences: List of prediction dicts, each with 'output', 'answer'.
    Returns:
        metrics_list: List of per-example metric dicts.
        metrics_agg: Dict of averaged metrics over the dataset.
    """
    metrics_list: List[Dict[str, float]] = []
    metrics_agg: Dict[str, float] = {"em": 0.0, "cover_match": 0.0, "str_f1": 0.0}

    for item, seq in zip(data, all_sequences):
        pred = seq.get('answer', "")
        gold_answers = item.get("golden_answers", [])
        if not pred:
            em = cover_match = str_f1 = 0.0
        else:
            norm_pred = normalize_answer_qa(pred)
            norm_gold = [normalize_answer_qa(g) for g in gold_answers]
            em = float(norm_pred in norm_gold)
            cover_match = float(any(g in norm_pred for g in norm_gold))
            str_f1 = max([string_f1(norm_pred, g) for g in norm_gold]) if norm_gold else 0.0
        metrics_agg["em"] += em
        metrics_agg["cover_match"] += cover_match
        metrics_agg["str_f1"] += str_f1
        metrics_list.append({"em": em, "cover_match": cover_match, "str_f1": str_f1})

    n = float(len(data)) if data else 1.0
    for k in metrics_agg:
        metrics_agg[k] /= n
    return metrics_list, metrics_agg

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input data (JSON file)")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        outputs = json.load(f)

    data = [
        {"question": item["question"], "golden_answers": item["gold"]}
        for item in outputs
    ]
    for item in outputs:
        item['finished'] = float(item.get("pred", "") != "")
        item['answer'] = item.get("pred", "")

    metric_list, metrics_agg = run_evaluation(data, outputs)
    print(metrics_agg)
    