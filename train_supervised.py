import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = "/scratch/yl9038/models/qwen3"  # Adjust as needed
DATA_PATH = "data/train_trajectory.jsonl"  # Update with your data path
OUTPUT_DIR = "./output/qwen3_sft_checkpoints"
IGNORE_INDEX = -100

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.samples = []
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                question = item["question"]
                trajectory = item["trajectory"]
                input_ids, labels = self.encode_and_mask(question, trajectory, max_length)
                self.samples.append({"input_ids": input_ids, "labels": labels})

    def encode_and_mask(self, question, trajectory, max_length):
        full_text = question + "\n" + trajectory
        mask = [False] * len(full_text)
        start = 0
        while True:
            s = full_text.find("<information>", start)
            if s == -1:
                break
            e = full_text.find("</information>", s)
            if e == -1:
                break
            for i in range(s, e + len("</information>")):
                mask[i] = True
            start = e + len("</information>")
        enc = self.tokenizer(full_text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        labels = enc.input_ids.copy()
        for idx, (start, end) in enumerate(enc.offset_mapping):
            if any(mask[start:end]):
                labels[idx] = IGNORE_INDEX
        return torch.tensor(enc.input_ids), torch.tensor(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)

    dataset = TrajectoryDataset(DATA_PATH, tokenizer)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main() 