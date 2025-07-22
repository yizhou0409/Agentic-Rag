#!/usr/bin/env python3
"""
Script to run LlamaFactory training on trajectory data.
This script:
1. Converts trajectory data to LlamaFactory format
2. Runs LlamaFactory training
3. Focuses on learning from trajectories without caring about answer correctness
"""

import os
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log the output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Success: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

def convert_data_to_llamafactory(input_files, output_dir):
    """Convert trajectory data to LlamaFactory format."""
    logger.info("Converting trajectory data to LlamaFactory format...")
    
    cmd = [
        "python", "convert_to_llamafactory.py",
        "--input_files"
    ] + input_files + [
        "--output_dir", output_dir
    ]
    
    run_command(cmd, "Data conversion")

def run_llamafactory_training(config_path, model_path):
    """Run LlamaFactory training."""
    logger.info("Starting LlamaFactory training...")
    
    # Check if LlamaFactory is installed
    try:
        import llamafactory
        logger.info("LlamaFactory is available")
    except ImportError:
        logger.error("LlamaFactory not found. Please install it first:")
        logger.error("pip install llamafactory")
        raise
    
    cmd = [
        "llamafactory-cli", "train",
        "--config", config_path,
        "--model_name_or_path", model_path
    ]
    
    run_command(cmd, "LlamaFactory training")

def main():
    parser = argparse.ArgumentParser(description="Run LlamaFactory training on trajectory data")
    parser.add_argument("--input_files", nargs="+", 
                       default=["data/hotpotqa/trajectory_train_5.jsonl", "data/2wikimultihop/trajectory_train_5.jsonl"],
                       help="Input trajectory files")
    parser.add_argument("--model_path", default="/scratch/yl9038/models/Qwen3-8B-Base",
                       help="Path to the base model")
    parser.add_argument("--config_path", default="llamafactory_config.yaml",
                       help="Path to LlamaFactory config file")
    parser.add_argument("--output_dir", default="data/llamafactory",
                       help="Output directory for converted data")
    parser.add_argument("--skip_conversion", action="store_true",
                       help="Skip data conversion step")
    
    args = parser.parse_args()
    
    # Validate inputs
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    # Step 1: Convert data to LlamaFactory format
    if not args.skip_conversion:
        convert_data_to_llamafactory(args.input_files, args.output_dir)
    else:
        logger.info("Skipping data conversion step")
    
    # Step 2: Run LlamaFactory training
    run_llamafactory_training(args.config_path, args.model_path)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 