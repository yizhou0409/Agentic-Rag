import os
import json
import subprocess
import glob
import time
import argparse

# Parse command-line arguments for N
parser = argparse.ArgumentParser(description="Prepare supervised trajectories for training.")
parser.add_argument('--n', type=int, default=5, help='Number of cases to extract and process (default: 200)')
args = parser.parse_args()
N = args.n
print(f"Number of cases to process: {N}")

# Paths
hotpotqa_in = "data/hotpotqa/train.jsonl"
hotpotqa_dir = "data/hotpotqa/hotpotqa"
hotpotqa_200 = os.path.join(hotpotqa_dir, f"train_{N}.jsonl")
multihop_in = "data/2wikimultihop/queries.jsonl"
multihop_dir = "data/2wikimultihop/2wikimultihop"
multihop_200 = os.path.join(multihop_dir, f"queries_{N}.jsonl")
config_path = "configs/inference.yaml"  # Adjust if needed
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(hotpotqa_dir, exist_ok=True)
os.makedirs(multihop_dir, exist_ok=True)

# Helper: extract first N lines from a JSONL file
def extract_first_n_jsonl(in_path, out_path, n=N, dataset_name=None):
    print(f"Extracting first {n} cases from {in_path} to {out_path}")
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            item = json.loads(line)
            if dataset_name == "2wikimultihop":
                item["question"] = item.get("text", "")
                answers = item.get("metadata", {}).get("answer", [])
                if not isinstance(answers, list):
                    answers = [answers]
                item["golden_answers"] = answers
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

# Helper: run rar_inference.py on a dataset
def run_rar_inference(subset_path, dataset_name, dataset_split, dataset_root_dir):
    print(f"Running rar_inference.py on {subset_path}")
    # Use hydra config overrides to set dataset path and output dir
    cmd = [
        "python", "rar_inference.py",
        f"data.dataset_root_dir={dataset_root_dir}",
        f"data.dataset_name={dataset_name}",
        f"data.dataset_split={dataset_split}",
        f"inference.max_turns=5"
    ]
    subprocess.run(cmd, check=True)
    # Wait a moment for output file to be written
    time.sleep(2)
    # Find the latest output json file recursively
    output_files = sorted(glob.glob("**/output_*.json", recursive=True), key=os.path.getmtime, reverse=True)
    if not output_files:
        raise FileNotFoundError("No output_*.json file found in any directory after inference.")
    latest_output = output_files[0]
    print(f"Found output file: {latest_output}")
    return latest_output

# Helper: reformat output for supervised training
def reformat_for_supervised(inference_output_path, supervised_out_path, dataset_name):
    print(f"Reformatting {inference_output_path} to {supervised_out_path}")
    with open(inference_output_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    with open(supervised_out_path, 'w', encoding='utf-8') as fout:
        for item in data:
            # The format from rar_inference.py is unified.
            question = item.get("question")
            # The 'answer' for training should be the ground truth from the 'gold' field.
            gold_answer = item.get("gold", [])
            if isinstance(gold_answer, list):
                answer_str = "; ".join(gold_answer)
            else:
                answer_str = gold_answer if gold_answer is not None else ""

            out = {
                "question": question,
                "trajectory": item.get("output"),
                "answer": answer_str,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} items to {supervised_out_path}")


def main():
    # 1. Extract first N cases
    extract_first_n_jsonl(hotpotqa_in, hotpotqa_200, N, dataset_name="hotpotqa")
    extract_first_n_jsonl(multihop_in, multihop_200, N, dataset_name="2wikimultihop")

    # 2. Run inference and reformat for hotpotqa
    hotpotqa_outfile = run_rar_inference(hotpotqa_200, "hotpotqa", f"train_{N}", "data/hotpotqa")
    reformat_for_supervised(hotpotqa_outfile, f"data/hotpotqa/trajectory_train_{N}.jsonl", "hotpotqa")

    # 3. Run inference and reformat for 2wikimultihop
    multihop_outfile = run_rar_inference(multihop_200, "2wikimultihop", f"queries_{N}", "data/2wikimultihop")
    reformat_for_supervised(multihop_outfile, f"data/2wikimultihop/trajectory_train_{N}.jsonl", "2wikimultihop")

    print("All done!")

if __name__ == "__main__":
    main() 