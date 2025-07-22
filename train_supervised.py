import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import logging
from glob import glob

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048, mask_retrieved=True):
        """
        Args:
            data_path: Path to the jsonl file containing trajectories
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            mask_retrieved: Whether to mask out retrieved information
        """
        self.samples = []
        self.tokenizer = tokenizer
        logger.info(f"Loading data from {data_path}")
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                question = item["question"]
                trajectory = item["trajectory"]
                if question is None or trajectory is None:
                    continue
                input_ids, labels = self.encode_and_mask(question, trajectory, max_length, mask_retrieved)
                self.samples.append({"input_ids": input_ids, "labels": labels})
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def encode_and_mask(self, question, trajectory, max_length, mask_retrieved):
        # Combine question and trajectory
        full_text = question + "\n" + trajectory
        
        # Create masking array (True for tokens to be masked)
        mask = [False] * len(full_text)
        
        # Mask out retrieved information if requested
        if mask_retrieved:
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

        # Encode text and apply masking
        enc = self.tokenizer(full_text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        labels = enc.input_ids.copy()
        
        # Apply masking to labels
        for idx, (start, end) in enumerate(enc.offset_mapping):
            if any(mask[start:end]):
                labels[idx] = IGNORE_INDEX

        return torch.tensor(enc.input_ids), torch.tensor(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on trajectory data")
    parser.add_argument("--model_path", type=str, default="/scratch/yl9038/models/Qwen3-0.6B",
                       help="Path to the pretrained model")
    parser.add_argument("--data_paths", type=str, nargs="+",default=,
                       help="Paths to the trajectory data files")
    parser.add_argument("--output_dir", type=str, default="./output/sft_checkpoints",
                       help="Directory to save the model checkpoints")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--mask_retrieved", action="store_true",
                       help="Whether to mask out retrieved information during training")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Training arguments: {args}")

    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Load all datasets
    datasets = []
    for data_path in args.data_paths:
        dataset = TrajectoryDataset(
            data_path, 
            tokenizer, 
            max_length=args.max_length,
            mask_retrieved=args.mask_retrieved
        )
        datasets.append(dataset)
    
    # Combine datasets if multiple
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
        logger.info(f"Combined {len(dataset)} samples from {len(datasets)} datasets")
    else:
        dataset = datasets[0]

    # Split into train and eval
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])
    logger.info(f"Train size: {train_size}, Eval size: {test_size}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        learning_rate=args.learning_rate,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    # Initialize trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train and save
    logger.info("Starting training")
    trainer.train()
    
    logger.info(f"Saving model and tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 