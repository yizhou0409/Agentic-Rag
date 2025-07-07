import re

from utils import normalize_answer_qa, string_f1

def run_evaluation(data, all_sequences):
    # initialize the metrics
    metrics_list = []
    metrics_agg = {
        "em": 0,
        "cover_match": 0,
        "str_f1": 0,
        "valid": 0,
        "finished": 0,
        "num_retrieval": 0,
        "num_fail_retrieval": 0,
    }

    # iterate over the data and the sequences
    for item, seq in zip(data, all_sequences):
        output = seq["output"]
        pred = seq['answer']
        question = item['question']
        gold_answer = item["golden_answers"] # a list of acceptable answers

        # if the prediction is empty, set the metric to 0 and the valid flag to 0 to indicate the invalidality
        if pred == "":
            em, cover_match, str_f1, valid = 0, 0, 0, 0

        # if the prediction is not empty, evluate with the metrics
        else:
            norm_pred = normalize_answer_qa (pred)
            norm_gold = [normalize_answer_qa(g) for g in gold_answer]

            # evaluate the exact match

            # if the prediction is directly in the gold answer, set the em to 1
            em = int(norm_pred in norm_gold)

            # if the prediction is longer but contain the exactly same information in the gold answer, set the cover_match to 1
            cover_match = int(any([g in norm_pred for g in norm_gold]))

            # evaluate the string f1 score, to avoid the case that the predcition is not the same since some words are not the same, but f1 score is still high, we evaluate it as pass the test
            str_f1 = max([string_f1(norm_pred, g) for g in norm_gold])
            
            # set the valid flag to 1 to indicate that the prediction is valid and can be used for further evaluation
            valid = 1

        # count the number of search queries
        matches = re.findall(r"<search>.*?</search>", output, flags=re.DOTALL)
        num_search = len(matches)

        # count the number of failed retrievals
        matches = re.findall(r"<information>\s*No helpful information found\s*</information>", output, flags=re.DOTALL)
        num_fail = len(matches)

        # update the metrics aggrations for the current item
        metrics_agg["num_retrieval"] += num_search
        metrics_agg["num_fail_retrieval"] += num_fail
        metrics_agg["em"] += em
        metrics_agg["cover_match"] += cover_match
        metrics_agg["str_f1"] += str_f1
        metrics_agg["valid"] += valid
        metrics_agg["finished"] += int(seq['finished'])

        # append the metrics to the list
        metrics_list.append(
            {
                "em": em,
                "cover_match": cover_match,
                "str_f1": str_f1,
                "valid": valid,
                "finished": int(seq['finished']),
                "num_retrieval": num_search,
                "num_fail_retrieval": num_fail,
            }
        )
    
    # calculate the average metrics
    for k in metrics_agg.keys():
        metrics_agg[k] /= len(data)

    # return the metrics list and the average metrics as the evaluation results
    return metrics_list, metrics_agg

def run_evaluation_simple(data, all_sequences):
    # initialize the metrics
    metrics_list = []
    metrics_agg = {
        "em": 0,
        "cover_match": 0,
        "str_f1": 0,
    }

    # iterate over the data and the sequences
    for item, seq in zip(data, all_sequences):
        output = seq["output"]
        pred = seq['answer']
        question = item['question']
        gold_answer = item["golden_answers"] # a list of acceptable answers

        # if the prediction is empty, set the metric to 0 and the valid flag to 0 to indicate the invalidality
        if pred == "":
            em, cover_match, str_f1 = 0, 0, 0

        # if the prediction is not empty, evluate with the metrics
        else:
            norm_pred = normalize_answer_qa (pred)
            norm_gold = [normalize_answer_qa(g) for g in gold_answer]

            # evaluate the exact match

            # if the prediction is directly in the gold answer, set the em to 1
            em = int(norm_pred in norm_gold)

            # if the prediction is longer but contain the exactly same information in the gold answer, set the cover_match to 1
            cover_match = int(any([g in norm_pred for g in norm_gold]))

            # evaluate the string f1 score, to avoid the case that the predcition is not the same since some words are not the same, but f1 score is still high, we evaluate it as pass the test
            str_f1 = max([string_f1(norm_pred, g) for g in norm_gold])

        # update the metrics aggrations for the current item
        metrics_agg["em"] += em
        metrics_agg["cover_match"] += cover_match
        metrics_agg["str_f1"] += str_f1

        # append the metrics to the list
        metrics_list.append(
            {
                "em": em,
                "cover_match": cover_match,
                "str_f1": str_f1,
            }
        )
    
    # calculate the average metrics
    for k in metrics_agg.keys():
        metrics_agg[k] /= len(data)

# test the functions
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)

    args = parser.parse_args()

    args = parser.parse_args()

    with open(args.output, "r") as f:
        outputs = json.load(f)

    data = [
        {
            "question": item["question"],
            "golden_answers": item["gold"],
        }
        for item in outputs
    ]

    for item in outputs:
        item['finished'] = int(item["pred"] != "")
        item['answer'] = item["pred"]

    metric_list, metrics_agg = run_evaluation(data, outputs)

    print(metrics_agg)
    