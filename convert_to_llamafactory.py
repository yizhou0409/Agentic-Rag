import json
import os
import argparse
from typing import List, Dict, Any

def convert_trajectory_to_llamafactory_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert trajectory data to LlamaFactory format.
    
    LlamaFactory expects:
    {
        "instruction": "question",
        "input": "",
        "output": "trajectory"
    }
    """
    question = item.get("question", "")
    trajectory = item.get("trajectory", "")
    
    # LlamaFactory format
    return {
        "instruction": question,
        "input": "",
        "output": trajectory
    }

def convert_file(input_path: str, output_path: str):
    """Convert a single trajectory file to LlamaFactory format."""
    converted_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                converted_item = convert_trajectory_to_llamafactory_format(item)
                converted_data.append(converted_item)
    
    # Save in LlamaFactory format
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_data)} samples from {input_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert trajectory data to LlamaFactory format")
    parser.add_argument("--input_files", nargs="+", required=True,
                       help="Input trajectory files to convert")
    parser.add_argument("--output_dir", default="data/llamafactory",
                       help="Output directory for converted files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert each file
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} does not exist, skipping...")
            continue
            
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}.json")
        
        convert_file(input_file, output_file)

if __name__ == "__main__":
    main() 