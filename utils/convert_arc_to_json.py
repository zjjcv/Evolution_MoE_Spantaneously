"""
Convert AI2 ARC (ARC-Challenge/ARC-Easy) parquet files to plain text JSON format.

Input: parquet files with columns: id, question, choices (dict with text/label), answerKey
Output: JSON files with choices formatted as plain text (e.g., "A. choice1\nB. choice2\n...")

Usage:
    python utils/convert_arc_to_json.py --data_dir data/ai2_arc/ARC-Challenge --output_dir data/ai2_arc/ARC-Challenge_json
    python utils/convert_arc_to_json.py --data_dir data/ai2_arc/ARC-Easy --output_dir data/ai2_arc/ARC-Easy_json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def format_choices(choices_dict: dict) -> str:
    """
    Format choices from dict to plain text.

    Args:
        choices_dict: Dict with 'text' (list of choice texts) and 'label' (list of labels, e.g., ['A', 'B', 'C', 'D'])

    Returns:
        Formatted string like "A. choice1\nB. choice2\nC. choice3\nD. choice4"
    """
    labels = choices_dict['label']
    texts = choices_dict['text']

    # Handle both numpy arrays and lists
    if hasattr(labels, 'tolist'):
        labels = labels.tolist()
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()

    lines = []
    for label, text in zip(labels, texts):
        lines.append(f"{label}. {text}")

    return "\n".join(lines)


def convert_parquet_to_json(parquet_path: Path, output_path: Path) -> None:
    """
    Convert a single parquet file to JSON.

    Args:
        parquet_path: Path to input parquet file
        output_path: Path to output JSON file
    """
    df = pd.read_parquet(parquet_path)

    records = []
    for _, row in df.iterrows():
        record = {
            "id": row['id'],
            "question": row['question'],
            "choices": format_choices(row['choices']),
            "answerKey": row['answerKey']
        }
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted {parquet_path.name} -> {output_path} ({len(records)} records)")


def main():
    parser = argparse.ArgumentParser(description="Convert ARC parquet files to plain text JSON")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ARC data directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory for JSON files")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Find all parquet files
    parquet_files = list(data_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    # Convert each parquet file
    for parquet_path in parquet_files:
        output_path = output_dir / f"{parquet_path.stem}.json"
        convert_parquet_to_json(parquet_path, output_path)

    print(f"\nConversion complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
