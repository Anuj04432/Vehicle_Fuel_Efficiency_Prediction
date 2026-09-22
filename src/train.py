"""
train.py
Trains, benchmarks, and persists the three professor-specified regression models:
1. Linear Regression (Baseline)
2. Support Vector Regressor (SVR - Non-linear Kernel)
3. Random Forest Regressor (Ensemble Bagging - Top Performer)
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cleaned_vehicles.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fuel_model_pipeline.joblib')


def simplify_transmission(trany: str) -> str:
    if not isinstance(trany, str):
        return 'Automatic'
    s = trany.lower()
    if 'manual' in s:
        return 'Manual'
    elif 'cvt' in s or 'variable' in s:
        return 'CVT'
    else:
        return 'Automatic'


def simplify_drive(drive: str) -> str:
    if not isinstance(drive, str):
        return 'Front-Wheel Drive'
    s = drive.lower()
    if 'front' in s:
        return 'Front-Wheel Drive'
    elif 'rear' in s:
        return 'Rear-Wheel Drive'
    elif 'all' in s or '4' in s:
        return 'All-Wheel / 4WD'
    return 'Front-Wheel Drive'


def simplify_vclass(vclass: str) -> str:
    if not isinstance(vclass, str):
        return 'Sedan / Passenger Car'
    s = vclass.lower()
    if 'sport utility' in s or 'suv' in s:
        return 'SUV'
    elif 'pickup' in s:
        return 'Pickup Truck'
    elif 'van' in s or 'minivan' in s:
        return 'Van / Minivan'
    elif 'two seater' in s:
        return 'Two Seater / Sports'
    elif 'station wagon' in s:
        return 'Station Wagon'
    else:
        return 'Sedan / Passenger Car'


def simplify_fuel(fuel: str) -> str:
    if not isinstance(fuel, str):
        return 'Regular Gasoline'
    s = fuel.lower()
    if 'diesel' in s:
        return 'Diesel'
    elif 'premium' in s:
        return 'Premium Gasoline'
    elif 'e85' in s or 'ethanol' in s:
        return 'Flex-Fuel (E85)'
    else:
        return 'Regular Gasoline'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['trans_type'] = data['trany'].apply(simplify_transmission)
    data['drive_simple'] = data['drive'].apply(simplify_drive)
    data['vclass_simple'] = data['VClass'].apply(simplify_vclass)
    data['fuel_simple'] = data['fuelType'].apply(simplify_fuel)
    data['displ_per_cylinder'] = data['displ'] / data['cylinders']
    return data


def train_and_export():
    print("=" * 65)
    print("STEP 1: PREPARING TRAINING DATASET")
    print("=" * 65)
    raw_df = pd.read_csv(DATA_PATH)
    data = engineer_features(raw_df)

    num_features = ['year', 'displ', 'cylinders', 'displ_per_cylinder', 'tCharger', 'sCharger', 'startStop']
    cat_features = ['trans_type', 'drive_simple', 'vclass_simple', 'fuel_simple']
    all_features = num_features + cat_features
    target = 'comb08'

    X = data[all_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Total Samples: {len(data)} | Training: {len(X_train)} | Testing: {len(X_test)}\n")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    models_dict = {}
    metrics_summary = {}

    # 1. Linear Regression
    print("Training 1/3: Linear Regression (Baseline)...")
    lr_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    models_dict['Linear Regression'] = lr_pipeline
    metrics_summary['Linear Regression'] = {
        'r2': float(r2_score(y_test, lr_preds)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, lr_preds))),
        'mae': float(mean_absolute_error(y_test, lr_preds))
    }

    # 2. Support Vector Regressor (SVR with RBF Kernel)
    print("Training 2/3: Support Vector Regressor (SVR with RBF Kernel)...")
    sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(10000, len(X_train)), replace=False)
    svr_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', SVR(C=10.0, epsilon=0.2, kernel='rbf'))])
    svr_pipeline.fit(X_train.iloc[sample_idx], y_train.iloc[sample_idx])
    svr_preds = svr_pipeline.predict(X_test)
    models_dict['Support Vector Regressor (SVR)'] = svr_pipeline
    metrics_summary['Support Vector Regressor (SVR)'] = {
        'r2': float(r2_score(y_test, svr_preds)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, svr_preds))),
        'mae': float(mean_absolute_error(y_test, svr_preds))
    }

    # 3. Random Forest Regressor
    print("Training 3/3: Random Forest Regressor (Ensemble Bagging)...")
    rf_pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1))])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    models_dict['Random Forest Regressor'] = rf_pipeline
    metrics_summary['Random Forest Regressor'] = {
        'r2': float(r2_score(y_test, rf_preds)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, rf_preds))),
        'mae': float(mean_absolute_error(y_test, rf_preds))
    }

    print("\n" + "=" * 65)
    print("STEP 2: MODEL EVALUATION & COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Model Algorithm':<32} | {'R² Score':<10} | {'RMSE (MPG)':<12} | {'MAE (MPG)':<10}")
    print("-" * 72)
    for name, m in metrics_summary.items():
        print(f"{name:<32} | {m['r2']:.4f}     | {m['rmse']:.2f}         | {m['mae']:.2f}")

    # Persist bundle
    os.makedirs(MODEL_DIR, exist_ok=True)
    metadata = {
        'num_features': num_features,
        'cat_features': cat_features,
        'trans_types': sorted(data['trans_type'].unique().tolist()),
        'drive_types': sorted(data['drive_simple'].unique().tolist()),
        'vclasses': sorted(data['vclass_simple'].unique().tolist()),
        'fuel_types': sorted(data['fuel_simple'].unique().tolist()),
        'year_range': (int(data['year'].min()), int(data['year'].max())),
        'displ_range': (float(data['displ'].min()), float(data['displ'].max())),
        'metrics_summary': metrics_summary,
        'best_model': 'Random Forest Regressor'
    }

    bundle = {
        'models': models_dict,
        'pipeline': models_dict['Random Forest Regressor'],  # default pipeline
        'metadata': metadata
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSuccessfully exported trained models and metadata to:\n{MODEL_PATH}")


if __name__ == '__main__':
    train_and_export()
