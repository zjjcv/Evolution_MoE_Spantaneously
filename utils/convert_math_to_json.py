"""
Convert MATH parquet files to JSON format.

This script converts MATH dataset from parquet to JSON format.

Usage:
    python utils/convert_math_to_json.py
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
            "problem": row['problem'],
            "level": row['level'],
            "type": row['type'],
            "solution": row['solution']
        }
        records.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted {parquet_path} -> {output_path} ({len(records)} records)")


def split_by_level(df: pd.DataFrame) -> dict:
    """Split dataset by difficulty level.

    Args:
        df: Input dataframe

    Returns:
        Dictionary mapping level names to filtered dataframes
    """
    splits = {}
    for level in df['level'].unique():
        splits[level] = df[df['level'] == level]
    return splits


def split_by_type(df: pd.DataFrame) -> dict:
    """Split dataset by problem type.

    Args:
        df: Input dataframe

    Returns:
        Dictionary mapping type names to filtered dataframes
    """
    splits = {}
    for ptype in df['type'].unique():
        splits[ptype] = df[df['type'] == ptype]
    return splits


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Convert MATH parquet to JSON')
    parser.add_argument('--split', choices=['all', 'level', 'type'], default='all',
                        help='How to split the output: all (single file), level, or type')
    parser.add_argument('--input-dir', default='/data/zjj/Synergistic_Core/data/MATH',
                        help='Input directory containing parquet files')
    parser.add_argument('--output-dir', default='/data/zjj/Synergistic_Core/data/MATH/json',
                        help='Output directory for JSON files')
    args = parser.parse_args()

    # Find all parquet files
    input_path = Path(args.input_dir)
    parquet_files = list(input_path.glob('*.parquet'))

    if not parquet_files:
        print(f"Error: No parquet files found in {args.input_dir}")
        return

    # Combine all parquet files
    df_list = []
    for pq_file in sorted(parquet_files):
        print(f"Loading {pq_file}...")
        df_list.append(pd.read_parquet(pq_file))

    df = pd.concat(df_list, ignore_index=True)
    print(f"Total records: {len(df)}")

    # Print dataset info
    print(f"\nDataset Info:")
    print(f"  Total records: {len(df)}")
    print(f"  Levels: {sorted(df['level'].unique())}")
    print(f"  Types: {sorted(df['type'].unique())}")

    # Convert based on split option
    if args.split == 'all':
        # Single output file
        output_path = os.path.join(args.output_dir, 'MATH_all.json')
        os.makedirs(args.output_dir, exist_ok=True)

        records = []
        for _, row in df.iterrows():
            record = {
                "problem": row['problem'],
                "level": row['level'],
                "type": row['type'],
                "solution": row['solution']
            }
            records.append(record)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_path}")

    elif args.split == 'level':
        # Split by difficulty level
        os.makedirs(args.output_dir, exist_ok=True)
        level_splits = split_by_level(df)

        for level, level_df in level_splits.items():
            safe_name = level.replace(' ', '_').replace('/', '_')
            output_path = os.path.join(args.output_dir, f'MATH_{safe_name}.json')

            records = []
            for _, row in level_df.iterrows():
                record = {
                    "problem": row['problem'],
                    "level": row['level'],
                    "type": row['type'],
                    "solution": row['solution']
                }
                records.append(record)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(records)} records to {output_path}")

    elif args.split == 'type':
        # Split by problem type
        os.makedirs(args.output_dir, exist_ok=True)
        type_splits = split_by_type(df)

        for ptype, type_df in type_splits.items():
            safe_name = ptype.replace(' ', '_').replace('/', '_')
            output_path = os.path.join(args.output_dir, f'MATH_{safe_name}.json')

            records = []
            for _, row in type_df.iterrows():
                record = {
                    "problem": row['problem'],
                    "level": row['level'],
                    "type": row['type'],
                    "solution": row['solution']
                }
                records.append(record)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(records)} records to {output_path}")

    print(f"\nConversion complete! Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
