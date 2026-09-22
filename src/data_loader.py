"""
data_loader.py
Module for loading and filtering raw EPA fuel economy dataset.
"""

import os
import pandas as pd
import numpy as np


RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'vehicles.csv')
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cleaned_vehicles.csv')

# Key features relevant for fuel efficiency prediction
CORE_COLUMNS = [
    # Metadata
    'year',
    'make',
    'model',
    'VClass',
    # Engine Specifications
    'displ',
    'cylinders',
    'fuelType',
    'trany',
    'drive',
    'tCharger',
    'sCharger',
    'startStop',
    # Targets
    'comb08',
    'city08',
    'highway08'
]


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw EPA vehicles dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    
    print(f"Loading raw dataset from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def extract_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts core predictive features and performs initial cleaning:
    - Filters out pure electric vehicles (which lack displacement/cylinders).
    - Normalizes binary indicator flags (turbocharger, supercharger).
    - Cleans missing values in core mechanical features.
    """
    # Select available columns
    available_cols = [c for c in CORE_COLUMNS if c in df.columns]
    data = df[available_cols].copy()

    # Filter for standard internal combustion and hybrid vehicles (fuel-based engines with displacement)
    # Pure EVs have displ = 0 / NaN and cylinders = 0 / NaN
    data = data.dropna(subset=['displ', 'cylinders', 'comb08'])
    data = data[(data['displ'] > 0) & (data['cylinders'] > 0) & (data['comb08'] > 0)]

    # Standardize binary indicators
    data['tCharger'] = data['tCharger'].apply(lambda x: 1 if str(x).strip().upper() == 'T' else 0)
    data['sCharger'] = data['sCharger'].apply(lambda x: 1 if str(x).strip().upper() == 'S' else 0)
    
    # Standardize start/stop flag
    if 'startStop' in data.columns:
        data['startStop'] = data['startStop'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

    # Fill remaining categorical missing values with 'Unknown'
    for cat_col in ['trany', 'drive']:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].fillna('Unknown')

    # Convert numeric types
    data['year'] = data['year'].astype(int)
    data['cylinders'] = data['cylinders'].astype(int)
    data['displ'] = data['displ'].astype(float)
    data['comb08'] = data['comb08'].astype(float)

    print(f"Filtered to {len(data)} valid fuel-engine vehicles across {data.shape[1]} attributes.")
    return data


def save_processed_data(df: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> str:
    """Saves cleaned dataset to processed directory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")
    return output_path


if __name__ == '__main__':
    raw_df = load_raw_data()
    cleaned_df = extract_core_features(raw_df)
    save_processed_data(cleaned_df)
