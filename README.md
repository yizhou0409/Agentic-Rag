# LongCoT-RAG-Inference

## Overview

This codebase implements a Reasoning-Augmented Retrieval (RAR) pipeline for multi-hop question answering, as well as supervised fine-tuning (SFT) for LLMs using both baseline and teacher-trajectory data. It leverages the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) toolkit for efficient retrieval and supports both open-source and OpenAI models for reasoning and summarization.

---

## 1. RAR Inference Pipeline

### Algorithm Summary

The RAR pipeline works as follows:
1. **Initial Prompting:** For each question, a prompt is constructed using a user/system template.
2. **Reasoning Step:** The LLM generates a reasoning step, which may include a search query.
3. **Retrieval:** If a search query is generated, relevant documents are retrieved from a corpus using FlashRAG (BM25 or dense retrieval).
4. **Information Integration:** Retrieved information is summarized and incorporated into the next reasoning step.
5. **Iterative Reasoning:** Steps 2–4 repeat for a fixed number of turns or until an answer is produced.
6. **Output:** The final answer and reasoning trace are saved, along with retrieval and evaluation metrics.

### File Structure

- `rar_inference.py`: Main script for running the RAR pipeline.
- `configs/inference.yaml`: Configuration file for inference (dataset, model, retrieval, etc).
- `FlashRAG/`: FlashRAG toolkit for retrieval.

### How to Run RAR Inference

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Otherwise you may refer to environment.yaml

2. **Prepare the retrieval index and corpus:**
   - Make sure your retrieval index (e.g., BM25 or FAISS) and corpus are built and paths are set in `configs/inference.yaml` under `retrieval.flashrag_config`.

3. **Edit the config file:**
   - Set the dataset, model paths, and retrieval settings in `configs/inference.yaml`.
   - Example:
     ```yaml
     data:
       dataset_name: hotpotqa
       dataset_split: dev
       dataset_root_dir: ./data
     model:
       reasoner_name_or_path: /path/to/your/reasoner/model
       summarizer_name_or_path: /path/to/your/summarizer/model
     retrieval:
       flashrag_config:
         retrieval_method: "bm25"
         index_path: /path/to/index
         corpus_path: /path/to/corpus.jsonl
     ```

4. **Run the inference pipeline:**
   ```bash
   python rar_inference.py
   ```
   - By default, this uses `configs/inference.yaml` via Hydra.
   - Outputs (reasoning traces, answers, metrics) will be saved in a Hydra-generated output directory (e.g., `outputs/2024-07-25/13-51-09/`).

5. **(Optional) Change config on the fly:**
   - You can override config options from the command line, e.g.:
     ```bash
     python rar_inference.py data.dataset_name=2wikimultihop
     ```

---

## 2. Supervised Fine-Tuning (SFT) for LLMs

### Data Preparation

- **Baseline Datasets:**  
  - `LLaMA-Factory/data/2wikimultihop_baseline_alpaca.json`
  - `LLaMA-Factory/data/hotpotqa_baseline_alpaca.json`
- **Teacher Trajectory Datasets:**  
  - `LLaMA-Factory/data/2wikimultihop_teacher_alpaca.json`
  - `LLaMA-Factory/data/hotpotqa_teacher_alpaca.json`
- All datasets are registered in `LLaMA-Factory/data/dataset_info.json`.

This project uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning (SFT) of language models. LLaMA-Factory provides a comprehensive framework for training, evaluation, and deployment of large language models.

---

## 5. Tips

- All configuration is managed via Hydra and YAML files for reproducibility.
- You can run inference and training on both baseline and teacher-trajectory data by simply changing the dataset names in the config or script.
- For more details on FlashRAG, see `FlashRAG/README.md`.

---

If you need a more detailed section on the RAR algorithm or want usage examples for other scripts, let me know! 