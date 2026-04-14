"""
Data Pipeline Module
Handles loading, cleaning, and preprocessing of the FakeNewsNet dataset.
"""

import os
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(filepath):
    """Load raw CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[DATA] Loaded {len(df)} rows from {filepath}")
    return df


def clean_data(df):
    """Clean the dataset by removing NAs and duplicates."""
    initial_count = len(df)

    df = df.dropna(subset=["title", "real"])
    after_na = len(df)
    print(f"[DATA] Dropped {initial_count - after_na} rows with missing values")

    df = df.drop_duplicates()
    after_dup = len(df)
    print(f"[DATA] Dropped {after_na - after_dup} duplicate rows")

    df = df.reset_index(drop=True)
    print(f"[DATA] Final dataset: {len(df)} rows")
    return df


def save_processed_data(df, filepath):
    """Save cleaned data to processed directory."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[DATA] Saved processed data to {filepath}")


def run_pipeline(config_path="config.yaml"):
    """Execute the full data pipeline."""
    config = load_config(config_path)

    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    print("=" * 50)
    print("DATA PIPELINE")
    print("=" * 50)

    df = load_raw_data(raw_path)
    df = clean_data(df)
    save_processed_data(df, processed_path)

    print("[DATA] Pipeline complete!")
    return df


if __name__ == "__main__":
    run_pipeline()
