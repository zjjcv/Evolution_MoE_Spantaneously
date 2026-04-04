"""
Convert GSM8K parquet files to JSON format.

This script converts GSM8K dataset from parquet to JSON format.

Usage:
    python utils/convert_gsm8k_to_json.py
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def convert_parquet_to_json(parquet_path: str, output_path: str) -> None:
    """Convert a single parquet file to JSON.

    Args:
        parquet_path: Path to input parquet file
        output_path: Path to output JSON file
    """
    df = pd.read_parquet(parquet_path)

    records = []
    for _, row in df.iterrows():
        record = {
            "question": row['question'],
            "answer": row['answer']
        }
        records.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted {parquet_path} -> {output_path} ({len(records)} records)")


def main():
    """Main execution function."""
    data_dir = "/data/zjj/Synergistic_Core/data/gsm8k/main"
    output_dir = "/data/zjj/Synergistic_Core/data/gsm8k/json"

    # Files to convert
    files = [
        ("test-00000-of-00001.parquet", "test.json"),
        ("train-00000-of-00001.parquet", "train.json"),
    ]

    for parquet_file, json_file in files:
        parquet_path = os.path.join(data_dir, parquet_file)
        output_path = os.path.join(output_dir, json_file)

        if os.path.exists(parquet_path):
            convert_parquet_to_json(parquet_path, output_path)
        else:
            print(f"Warning: File not found: {parquet_path}")

    print(f"\nConversion complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
