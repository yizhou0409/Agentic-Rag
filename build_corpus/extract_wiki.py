import argparse
import os
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Wikipedia XML dump to temp directory using WikiExtractor.")
    parser.add_argument("--dump_path", type=str, required=True, help="Path to the Wikipedia XML .bz2 dump file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of processes to use for extraction.")
    parser.add_argument("--output_dir", type=str, default="temp", help="Directory to store extracted files.")
    args = parser.parse_args()

    temp_dir = os.path.abspath(args.output_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if not os.path.exists(temp_dir) or len(os.listdir(temp_dir)) == 0:
        print(f"Extracting wiki dump to {temp_dir}...")
        subprocess.run([
            "python",
            "-m",
            "wikiextractor.WikiExtractor",
            "--json",
            "-o",
            temp_dir,
            "--process",
            str(args.num_workers),
            args.dump_path,
        ], check=True)
        print("Extraction complete.")
    else:
        print(f"Wiki dump already extracted in {temp_dir}, skipping extraction.")