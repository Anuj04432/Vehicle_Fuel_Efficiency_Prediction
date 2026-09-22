"""
eda.py
Exploratory Data Analysis script for Vehicle Fuel Efficiency dataset.
Computes statistical summaries, missing value profiles, and exports EDA visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cleaned_vehicles.csv')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'figures')


def run_eda(filepath: str = DATA_PATH):
    print("=" * 60)
    print("STEP 1: LOADING CLEANED DATASET FOR EDA")
    print("=" * 60)
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")

    print("=" * 60)
    print("STEP 2: DATASET OVERVIEW & MISSING VALUES")
    print("=" * 60)
    print(df.info())
    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\n" + "=" * 60)
    print("STEP 3: DESCRIPTIVE STATISTICS (NUMERICAL)")
    print("=" * 60)
    num_cols = ['year', 'displ', 'cylinders', 'comb08', 'city08', 'highway08']
    print(df[num_cols].describe().T)

    print("\nTarget (comb08 MPG) Skewness:", round(df['comb08'].skew(), 3))

    print("\n" + "=" * 60)
    print("STEP 4: CATEGORICAL DISTRIBUTIONS (TOP CATEGORIES)")
    print("=" * 60)
    for col in ['drive', 'fuelType', 'VClass']:
        print(f"\nValue Counts for {col} (Top 5):")
        print(df[col].value_counts().head(5))

    print("\n" + "=" * 60)
    print("STEP 5: GENERATING & SAVING EDA VISUALIZATIONS")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.1)

    # 1. Target Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(df['comb08'], bins=60, kde=True, color='royalblue', edgecolor='black')
    plt.title('Distribution of Combined Fuel Economy (comb08 MPG)', fontsize=14, fontweight='bold')
    plt.xlabel('Combined MPG (comb08)', fontsize=12)
    plt.ylabel('Vehicle Count', fontsize=12)
    plt.axvline(df['comb08'].median(), color='red', linestyle='--', label=f"Median: {df['comb08'].median():.1f} MPG")
    plt.axvline(df['comb08'].mean(), color='green', linestyle='-', label=f"Mean: {df['comb08'].mean():.1f} MPG")
    plt.legend()
    plt.tight_layout()
    dist_path = os.path.join(FIGURES_DIR, 'target_distribution.png')
    plt.savefig(dist_path, dpi=200)
    plt.close()
    print(f"[Saved] {dist_path}")

    # 2. Engine Displacement vs MPG Scatter Plot
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    scatter = sns.scatterplot(
        data=sample_df,
        x='displ',
        y='comb08',
        hue='cylinders',
        palette='viridis',
        alpha=0.6,
        s=30
    )
    plt.title('Engine Displacement (Liters) vs. Combined MPG', fontsize=14, fontweight='bold')
    plt.xlabel('Engine Displacement (L)', fontsize=12)
    plt.ylabel('Combined MPG', fontsize=12)
    plt.tight_layout()
    scatter_path = os.path.join(FIGURES_DIR, 'displacement_vs_mpg.png')
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"[Saved] {scatter_path}")

    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)
    plt.title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    print(f"[Saved] {heatmap_path}")

    # 4. MPG Distribution by Drive Type
    plt.figure(figsize=(10, 5))
    top_drives = df['drive'].value_counts().head(5).index
    sns.boxplot(
        data=df[df['drive'].isin(top_drives)],
        x='drive',
        y='comb08',
        palette='Set2'
    )
    plt.title('Combined MPG by Drive Type', fontsize=14, fontweight='bold')
    plt.xlabel('Drive Type', fontsize=12)
    plt.ylabel('Combined MPG', fontsize=12)
    plt.xticks(rotation=20)
    plt.tight_layout()
    box_path = os.path.join(FIGURES_DIR, 'mpg_by_drive.png')
    plt.savefig(box_path, dpi=200)
    plt.close()
    print(f"[Saved] {box_path}")

    print("\nEDA Completed successfully.")


if __name__ == '__main__':
    run_eda()
