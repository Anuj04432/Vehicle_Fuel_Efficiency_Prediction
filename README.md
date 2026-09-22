# 🚗 Real-World Vehicle Fuel Efficiency Prediction

An end-to-end Machine Learning project to predict vehicle fuel efficiency (Combined MPG and Metric L/100km) based on vehicle powertrain and technical specifications, featuring an interactive **Streamlit** web application.

---

## 📊 Dataset & Highlights
* **Data Source**: Certified US Environmental Protection Agency (EPA) Real-World Fuel Economy Dataset (`data/raw/vehicles.csv`).
* **Dataset Size**: **48,752 vehicles** (spanning models from 1984 through 2026).
* **Model Algorithms**: Benchmarking three classic ML paradigms:
  1. **Linear Regression** (Parametric baseline)
  2. **Support Vector Regressor — SVR** (Non-linear RBF kernel)
  3. **Random Forest Regressor** (Ensemble bagging trees — Top Performer)

### Model Evaluation Results (Test Set Comparison)
| Model Algorithm | $R^2$ Score | RMSE (MPG) | MAE (MPG) | Paradigm |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | `0.7669 (76.7%)` | `2.75 MPG` | `1.87 MPG` | Parametric Linear Baseline |
| **Support Vector Regressor (SVR)** | `0.8801 (88.0%)` | `1.97 MPG` | `1.30 MPG` | Non-Linear RBF Kernel |
| **Random Forest Regressor (Winner)** | **`0.9237 (92.4%)`** | **`1.57 MPG`** | **`0.98 MPG`** | Ensemble Bagging Trees |

---

## 📁 Project Structure

```
Vehicle_Fuel_Efficiency_Prediction/
├── app.py                      # Interactive Streamlit Web UI
├── data/
│   ├── raw/
│   │   └── vehicles.csv        # 50,000+ row raw EPA dataset
│   └── processed/
│       └── cleaned_vehicles.csv# Cleaned 48k+ vehicle dataset
├── models/
│   └── fuel_model_pipeline.joblib # Serialized model bundle & metadata
├── notebooks/
├── reports/
│   └── figures/                # EDA visualization charts
│       ├── correlation_heatmap.png
│       ├── displacement_vs_mpg.png
│       ├── mpg_by_drive.png
│       └── target_distribution.png
├── src/
│   ├── data_loader.py          # Data ingestion and raw column filtering
│   ├── eda.py                  # Exploratory Data Analysis & plot generator
│   └── train.py                # Feature engineering, training & model export
├── plan.md                     # Roadmap and milestone tracking
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 How to Run the Streamlit Application

1. **Activate your virtual environment**:
   ```powershell
   .\myvenv\Scripts\Activate.ps1
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`.

---

## 🌟 Streamlit UI Features
* **Interactive Vehicle Controls**: Sliders and selectors for Model Year, Engine Displacement (L), Cylinders, Transmission Type, Drivetrain (FWD/RWD/AWD), Fuel Type, and Turbo/Supercharger.
* **Quick Presets**: 1-click loading for common vehicles (e.g., Civic Compact Sedan, Midsize SUV, V8 Muscle Car, Pickup Truck).
* **Dual Output**: Displays both **Combined MPG** and **Metric L/100km** alongside estimated City/Highway figures.
* **Eco Rating**: Visual classification (Green / Moderate / High Consumption).
* **Annual Cost Calculator**: Computes estimated yearly fuel expenses based on annual mileage and gas prices.
* **Data Insights Tab**: View embedded exploratory analysis charts directly within the application.
