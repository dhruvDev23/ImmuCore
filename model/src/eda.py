"""
eda.py — Exploratory Data Analysis for ImmuCore
=================================================

This script runs a light but thorough EDA pass on the Pima Indians Diabetes
dataset. The goal is to give Tushar (Week 2 owner) everything he needs to
start modeling without redoing any of this setup work.

What this script covers:
  1. Basic statistics and data overview
  2. Missing value analysis — the dataset uses 0 as a placeholder for
     missing values in some columns (e.g., you can't have 0 blood pressure
     and be alive), so we flag those
  3. Class balance — how many diabetic vs not-diabetic samples
  4. Feature distributions — histogram for each feature
  5. Correlation matrix — which features relate to each other and to the target
  6. Box plots — to spot outliers at a glance

All charts are saved as PNG files in model/notebooks/charts/ so they can
be viewed without re-running the script.

Usage:
  Run from the project root:
    python model/src/eda.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend so it works without a display

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# -- Import our data loading function from load_data.py --
# We reuse the loader so there's one single source of truth for
# column names and file paths.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_diabetes_data

# Suppress some matplotlib warnings that clutter the output
warnings.filterwarnings("ignore", category=UserWarning)


def setup_chart_directory():
    """
    Create the charts output directory if it doesn't already exist.

    Charts go in model/notebooks/charts/ — this keeps them near the
    notebooks folder but separate from any .ipynb files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    charts_dir = os.path.join(project_root, "model", "notebooks", "charts")
    charts_dir = os.path.normpath(charts_dir)
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir


def analyze_missing_values(df):
    """
    Analyze "hidden" missing values in the dataset.

    The Pima dataset doesn't use NaN for missing values — instead, some
    columns have 0 where 0 is biologically impossible. For example:
      - Glucose: 0 is impossible for a living person
      - BloodPressure: 0 means no blood pressure (dead)
      - SkinThickness: 0 is technically possible but usually means "not recorded"
      - Insulin: 0 usually means "not measured"
      - BMI: 0 is impossible

    We flag these so Tushar knows which columns need imputation in Week 2.
    """

    print("\n" + "=" * 60)
    print("  MISSING VALUE ANALYSIS")
    print("  (Columns where 0 is biologically implausible)")
    print("=" * 60)

    # These are the columns where 0 almost certainly means "missing"
    # rather than an actual measurement of zero.
    suspicious_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    results = []
    for col in suspicious_columns:
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        results.append({
            "Column": col,
            "Zero Count": zero_count,
            "Zero %": round(zero_pct, 1),
        })
        status = "⚠️  Needs attention" if zero_pct > 5 else "✓ OK"
        print(f"  {col:30s} -> {zero_count:3d} zeros ({zero_pct:5.1f}%)  {status}")

    # -- Columns that are fine with zeros --
    ok_columns = ["Pregnancies", "Outcome"]
    print(f"\n  Columns where 0 is valid: {', '.join(ok_columns)}")
    print(f"  (Pregnancies = 0 just means no pregnancies, which is normal)")

    return pd.DataFrame(results)


def analyze_class_balance(df):
    """
    Check how balanced the target variable (Outcome) is.

    If the classes are heavily imbalanced, Tushar will need to handle
    this during modeling (e.g., using class weights or SMOTE).
    """

    print("\n" + "=" * 60)
    print("  CLASS BALANCE")
    print("=" * 60)

    counts = df["Outcome"].value_counts().sort_index()
    total = len(df)

    for label, count in counts.items():
        pct = (count / total) * 100
        tag = "Diabetic" if label == 1 else "Not Diabetic"
        bar = "█" * int(pct / 2)  # Simple visual bar
        print(f"  {label} ({tag:>12s}): {count:4d} ({pct:5.1f}%)  {bar}")

    ratio = counts[0] / counts[1]
    print(f"\n  Ratio (Not Diabetic : Diabetic): {ratio:.2f} : 1")

    if ratio > 3:
        print("  ⚠️  Significant imbalance — consider SMOTE or class weights")
    elif ratio > 1.5:
        print("  ⚠️  Moderate imbalance — class weights recommended")
    else:
        print("  ✓  Reasonably balanced")


def plot_feature_distributions(df, charts_dir):
    """
    Create a histogram for each feature, colored by the Outcome class.

    This shows how the feature values are distributed and whether there's
    visible separation between the diabetic and non-diabetic groups.
    """

    print("\n  Generating feature distribution plots...")

    features = [col for col in df.columns if col != "Outcome"]

    # We'll make a 2x4 grid of histograms — one per feature
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Feature Distributions by Outcome", fontsize=16, fontweight="bold", y=1.02)

    colors = ["#3498db", "#e74c3c"]  # Blue for not diabetic, red for diabetic
    labels = ["Not Diabetic (0)", "Diabetic (1)"]

    for i, feature in enumerate(features):
        row = i // 4
        col = i % 4
        ax = axes[row][col]

        # Plot overlapping histograms for each class
        for outcome_val in [0, 1]:
            subset = df[df["Outcome"] == outcome_val][feature]
            ax.hist(
                subset,
                bins=25,
                alpha=0.6,
                color=colors[outcome_val],
                label=labels[outcome_val],
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(charts_dir, "feature_distributions.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_correlation_matrix(df, charts_dir):
    """
    Create a heatmap showing correlations between all features and the target.

    Strong correlations with Outcome tell us which features are most predictive.
    Strong correlations between features might indicate multicollinearity.
    """

    print("  Generating correlation matrix...")

    corr_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,           # Show correlation values in each cell
        fmt=".2f",            # Two decimal places
        cmap="RdBu_r",       # Red-blue diverging colormap
        center=0,             # Center the colormap at 0
        square=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    filepath = os.path.join(charts_dir, "correlation_matrix.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")

    # Also print the correlations with the target, sorted by strength
    print("\n  Correlations with Outcome (sorted by absolute value):")
    target_corr = corr_matrix["Outcome"].drop("Outcome").abs().sort_values(ascending=False)
    for feature, corr_val in target_corr.items():
        direction = "+" if corr_matrix.loc[feature, "Outcome"] > 0 else "-"
        bar = "█" * int(corr_val * 20)
        print(f"    {feature:30s} {direction}{corr_val:.3f}  {bar}")


def plot_box_plots(df, charts_dir):
    """
    Create box plots for each feature, split by Outcome.

    Box plots are great for spotting outliers and seeing how the median
    values differ between the two classes.
    """

    print("  Generating box plots...")

    features = [col for col in df.columns if col != "Outcome"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Feature Box Plots by Outcome", fontsize=16, fontweight="bold", y=1.02)

    colors = {"0": "#3498db", "1": "#e74c3c"}

    for i, feature in enumerate(features):
        row = i // 4
        col = i % 4
        ax = axes[row][col]

        # Create a temporary column with string labels for prettier plotting
        temp_df = df[[feature, "Outcome"]].copy()
        temp_df["Class"] = temp_df["Outcome"].map({0: "Not Diabetic", 1: "Diabetic"})

        sns.boxplot(
            data=temp_df,
            x="Class",
            y=feature,
            hue="Class",
            ax=ax,
            palette=["#3498db", "#e74c3c"],
            width=0.5,
            fliersize=3,
            legend=False,
        )

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(charts_dir, "box_plots.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def print_detailed_stats(df):
    """
    Print detailed statistics for each feature — goes a step beyond
    the basic describe() output.
    """

    print("\n" + "=" * 60)
    print("  DETAILED FEATURE STATISTICS")
    print("=" * 60)

    features = [col for col in df.columns if col != "Outcome"]

    for feature in features:
        values = df[feature]
        print(f"\n  {feature}:")
        print(f"    Range:    {values.min():.1f} — {values.max():.1f}")
        print(f"    Mean:     {values.mean():.2f}")
        print(f"    Median:   {values.median():.2f}")
        print(f"    Std Dev:  {values.std():.2f}")
        print(f"    Skewness: {values.skew():.2f}")

        # Check for suspicious zeros
        zero_count = (values == 0).sum()
        if zero_count > 0 and feature not in ["Pregnancies", "Outcome"]:
            print(f"    Zeros:    {zero_count} (likely missing values)")


def main():
    """
    Main function — runs the full EDA pipeline.

    This is the entry point when you run the script directly. It loads
    the data, runs every analysis, saves the charts, and prints a summary
    at the end.
    """

    print("\n" + "=" * 60)
    print("  ImmuCore — Exploratory Data Analysis")
    print("  Pima Indians Diabetes Dataset")
    print("=" * 60)

    # -- Step 1: Load the data --
    print("\n[1/6] Loading dataset...")
    df = load_diabetes_data()
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns. ✓")

    # -- Step 2: Set up the charts directory --
    charts_dir = setup_chart_directory()
    print(f"  Charts will be saved to: {charts_dir}")

    # -- Step 3: Missing value analysis --
    print("\n[2/6] Analyzing missing values...")
    missing_df = analyze_missing_values(df)

    # -- Step 4: Class balance --
    print("\n[3/6] Checking class balance...")
    analyze_class_balance(df)

    # -- Step 5: Detailed stats --
    print("\n[4/6] Computing detailed statistics...")
    print_detailed_stats(df)

    # -- Step 6: Generate charts --
    print("\n[5/6] Generating charts...")
    plot_feature_distributions(df, charts_dir)
    plot_correlation_matrix(df, charts_dir)
    plot_box_plots(df, charts_dir)

    # -- Done --
    print("\n" + "=" * 60)
    print("  EDA COMPLETE")
    print("=" * 60)
    print(f"\n  Charts saved to: {charts_dir}")
    print("  Files generated:")
    print("    - feature_distributions.png")
    print("    - correlation_matrix.png")
    print("    - box_plots.png")

    print("\n  Key findings for Tushar (Week 2):")
    print("    1. Insulin and SkinThickness have lots of zeros (missing values)")
    print("       → Will need imputation or careful handling")
    print("    2. Class balance is moderate (~65/35 split)")
    print("       → Class weights are recommended but SMOTE may not be necessary")
    print("    3. Glucose has the strongest correlation with Outcome")
    print("       → Should be a strong predictor")
    print("    4. Some features have outliers (e.g., Insulin, Pregnancies)")
    print("       → Consider whether to clip or keep")
    print()


# -- Run the EDA when this script is executed directly --
if __name__ == "__main__":
    main()
