"""
Convert MMLU Parquet Files to JSON Format

This script converts MMLU dataset from parquet format to JSON format.
It processes all subtasks in the MMLU directory and creates JSON files
for train, validation, and test splits.

Author: Claude Code
Date: 2026-03-20
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm


# Configuration
MMLU_DATA_DIR = "/data/zjj/Synergistic_Core/data/mmlu"
OUTPUT_DIR = os.path.join(MMLU_DATA_DIR, "json_test")  # Save test sets to json_test subfolder

# Subtasks to process (all directories except "all" and ".cache")
SUBTASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "auxiliary_train",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def read_parquet_file(file_path: str) -> List[Dict[str, Any]]:
    """Read a parquet file and return as list of dictionaries.

    Args:
        file_path: Path to the parquet file

    Returns:
        List of dictionaries containing the data
    """
    df = pd.read_parquet(file_path)

    # Convert DataFrame to list of dictionaries
    data = []
    for _, row in df.iterrows():
        item = {
            "question": row["question"],
            "subject": row["subject"],
            "choices": row["choices"].tolist() if hasattr(row["choices"], 'tolist') else list(row["choices"]),
            "answer": int(row["answer"])
        }
        data.append(item)

    return data


def process_test_split(subtask: str) -> List[Dict[str, Any]]:
    """Process only the test split for a subtask.

    Args:
        subtask: Name of the subtask

    Returns:
        List of dictionaries containing the test data
    """
    subtask_dir = os.path.join(MMLU_DATA_DIR, subtask)
    split = "test"

    # Find the parquet file for test split
    parquet_files = [f for f in os.listdir(subtask_dir) if f.startswith(split) and f.endswith(".parquet")]

    if not parquet_files:
        print(f"  Warning: No {split} file found for {subtask}")
        return []

    # Read all parquet files for this split (usually just one)
    all_data = []
    for parquet_file in parquet_files:
        file_path = os.path.join(subtask_dir, parquet_file)
        data = read_parquet_file(file_path)
        all_data.extend(data)

    return all_data


def convert_subtask(subtask: str) -> List[Dict[str, Any]]:
    """Convert only the test split for a single subtask.

    Args:
        subtask: Name of the subtask

    Returns:
        List of dictionaries containing the test data
    """
    print(f"\nProcessing {subtask}...")

    # Process only test split
    test_data = process_test_split(subtask)
    print(f"  test: {len(test_data)} questions")

    return test_data


def save_test_json(subtask: str, test_data: List[Dict[str, Any]]) -> None:
    """Save the test split for a subtask to a JSON file.

    Args:
        subtask: Name of the subtask
        test_data: List containing the test data
    """
    output_file = os.path.join(OUTPUT_DIR, f"{subtask}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {output_file} ({len(test_data)} questions)")


def convert_all_subtasks() -> None:
    """Convert all MMLU subtasks test sets to JSON format."""
    print("=" * 60)
    print("MMLU Test Set to JSON Converter")
    print("=" * 60)
    print(f"Input directory: {MMLU_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Subtasks to process: {len(SUBTASKS)}")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each subtask
    for subtask in tqdm(SUBTASKS, desc="Converting subtasks"):
        subtask_dir = os.path.join(MMLU_DATA_DIR, subtask)

        # Check if subtask directory exists
        if not os.path.exists(subtask_dir):
            print(f"\nWarning: Directory not found for {subtask}, skipping...")
            continue

        # Convert the test set for this subtask
        test_data = convert_subtask(subtask)

        # Save to JSON
        if test_data:
            save_test_json(subtask, test_data)

    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)


def main():
    """Main function to run the conversion."""
    convert_all_subtasks()


if __name__ == "__main__":
    main()
