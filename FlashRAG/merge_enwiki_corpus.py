#!/usr/bin/env python3
"""merge_enwiki_jsonl.py

Utility to combine all compressed JSON dump pieces under
`enwiki-20171001-pages-meta-current-withlinks-processed/` into one
newline-delimited JSON (jsonl) file with the simplified schema::

    {"id": <page id>, "contents": <full page text>}

The script streams each bz2 file line-by-line, so memory usage stays low
and no temporary uncompressed files are created.

Usage
-----
python merge_enwiki_jsonl.py \
       --input-dir enwiki-20171001-pages-meta-current-withlinks-processed \
       --output-file enwiki_all.jsonl  

The default values above will be used when the flags are omitted.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List
from html import unescape
import re


def iter_bz2_files(root: Path) -> Iterable[Path]:
    """Yield all ``*.bz2`` files below *root* in lexicographic order."""
    for path in sorted(root.glob("**/*.bz2")):
        if path.is_file():
            yield path


def extract_contents(text_field):
    """Return a plain string from the ``text`` value in the source JSON.

    According to the processed dump format, the *text* field can be a
    list (paragraphs) or already a string. This helper normalises both
    cases by joining list elements with single spaces.
    """
    if isinstance(text_field, list):
        text = " ".join(map(str, text_field))
    else:
        text = str(text_field)

    # --- basic HTML cleanup -------------------------------------------------
    # 1. Replace common HTML entities (e.g. &amp; &lt;)
    text = unescape(text)

    # 2. Strip any residual HTML tags (e.g. <br>, <ref>, <p>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse excessive whitespace introduced by removals
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_file(path: Path, out_fh):
    """Stream *path*, writing one simplified JSON object per line to *out_fh*."""
    with bz2.open(path, mode="rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue  # skip empty lines (should not happen)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                # Corrupted line – report and skip.
                print(f"[warn] JSON decode error in {path}: {exc}", file=sys.stderr)
                continue
            simplified = {
                "id": obj.get("id"),
                "contents": extract_contents(obj.get("text", "")),
            }
            out_fh.write(json.dumps(simplified, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge enwiki dump pieces into a single jsonl file.")
    parser.add_argument("--input-dir", type=Path,
                        default=Path("enwiki-20171001-pages-meta-current-withlinks-processed"),
                        help="Root directory containing sub-directories with *.bz2 wiki_XX files.")
    parser.add_argument("--output-file", type=Path, default=Path("enwiki_all.jsonl"),
                        help="Destination jsonl file path.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel worker threads (default: number of CPUs).")
    parser.add_argument("--chunk-size", type=int, default=10000,
                        help="Number of documents to buffer in memory before flushing to disk (helps avoid OOM).")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory '{args.input_dir}' does not exist or is not a directory.")

    # Ensure parent directory for output exists.
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_docs = 0

    print(f"Scanning for .bz2 files under '{args.input_dir}' …", file=sys.stderr)
    files: List[Path] = list(iter_bz2_files(args.input_dir))
    if not files:
        parser.error("No .bz2 files found – check input directory.")

    print(f"Found {len(files)} files. Starting merge with {args.workers} worker(s)…", file=sys.stderr)

    # Use tqdm for a single overall progress bar (per file).
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:  # pragma: no cover
        print("[warn] tqdm not found; install via `pip install tqdm` for a nicer progress bar.", file=sys.stderr)

        def _tqdm(iterable, **kwargs):
            return iterable  # fall back to plain iterator

        tqdm = _tqdm  # type: ignore

    write_lock = threading.Lock()

    with args.output_file.open("w", encoding="utf-8") as out_fh:

        def _worker(p: Path):
            nonlocal total_docs
            buf: List[str] = []
            with bz2.open(p, mode="rt", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    simplified = {
                        "id": obj.get("id"),
                        "contents": extract_contents(obj.get("text", "")),
                    }
                    buf.append(json.dumps(simplified, ensure_ascii=False))

                    # Flush if buffer exceeds the configured chunk size to keep memory usage bounded
                    if len(buf) >= args.chunk_size:
                        with write_lock:
                            out_fh.write("\n".join(buf) + "\n")
                        buf.clear()

            # Write any remaining buffered lines
            if buf:
                with write_lock:
                    out_fh.write("\n".join(buf) + "\n")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(_worker, files), total=len(files), unit="file"))

    print(f"Done. Merged data written to '{args.output_file}'.", file=sys.stderr)


if __name__ == "__main__":
    main() 