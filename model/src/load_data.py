"""
load_data.py — Data Loading Script for ImmuCore
=================================================

This script loads the Pima Indians Diabetes dataset from the local CSV file
and prints a quick summary so we can confirm everything looks right before
moving on to EDA or modeling.

The CSV file we downloaded doesn't have a header row, so we manually assign
the column names here. These names match the standard Kaggle version of
this dataset exactly.

What this script does:
  1. Reads the CSV from model/data/diabetes.csv
  2. Assigns proper column names (the raw file has no header)
  3. Prints the shape, first few rows, data types, and basic stats
  4. Checks for any obvious problems (nulls, unexpected types)

Usage:
  Run from the project root:
    python model/src/load_data.py
"""

import os
import pandas as pd


def load_diabetes_data():
    """
    Load the Pima Indians Diabetes dataset from the local CSV file.

    The raw CSV has no header row, so we define the column names manually.
    These are the 8 input features plus the binary target column (Outcome).

    Returns:
        pandas.DataFrame: The full dataset with proper column names.
    """

    # -- Figure out the path to the CSV file --
    # We use os.path to build the path relative to this script's location,
    # so it works no matter where you run the command from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    csv_path = os.path.join(project_root, "model", "data", "diabetes.csv")
    csv_path = os.path.normpath(csv_path)

    # -- Define column names --
    # The dataset has 8 features and 1 target. These names match the
    # standard Kaggle version so Tushar can reference them directly.
    column_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",  # Target: 1 = diabetic, 0 = not diabetic
    ]

    # -- Load the CSV --
    # header=None tells pandas there's no header row in the file.
    # We pass our own column names instead.
    df = pd.read_csv(csv_path, header=None, names=column_names)

    return df


def print_summary(df):
    """
    Print a human-readable summary of the loaded dataset.

    This covers the basics: shape, first rows, types, stats, and null check.
    It's meant to be a quick sanity check, not a deep analysis — that's
    what eda.py is for.
    """

    print("=" * 60)
    print("  ImmuCore — Dataset Summary")
    print("  Pima Indians Diabetes Database")
    print("=" * 60)

    # -- Shape --
    print(f"\nRows: {df.shape[0]}   |   Columns: {df.shape[1]}")

    # -- First 5 rows --
    print("\n--- First 5 Rows ---")
    print(df.head().to_string(index=False))

    # -- Data types --
    print("\n--- Data Types ---")
    for col in df.columns:
        print(f"  {col:30s} -> {df[col].dtype}")

    # -- Basic statistics --
    print("\n--- Basic Statistics ---")
    print(df.describe().round(2).to_string())

    # -- Null check --
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n--- Null Values ---")
    if total_nulls == 0:
        print("  No null values found. ✓")
    else:
        print(f"  Total nulls: {total_nulls}")
        for col, count in null_counts.items():
            if count > 0:
                print(f"    {col}: {count}")

    # -- Class balance --
    print("\n--- Class Balance (Outcome) ---")
    counts = df["Outcome"].value_counts()
    total = len(df)
    for label, count in counts.items():
        pct = (count / total) * 100
        tag = "Diabetic" if label == 1 else "Not Diabetic"
        print(f"  {label} ({tag}): {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("  Dataset loaded successfully. Ready for EDA.")
    print("=" * 60)


# -- Main entry point --
# When you run this script directly, it loads the data and prints the summary.
# Other scripts (like eda.py) can import the load function without triggering
# the print.
if __name__ == "__main__":
    df = load_diabetes_data()
    print_summary(df)
