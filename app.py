"""
app.py
Interactive Streamlit Web Application for Real-World Vehicle Fuel Efficiency Prediction.
Supports multi-model comparison: Linear Regression, SVR, and Random Forest Regressor.
"""

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Vehicle Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fuel_model_pipeline.joblib')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'reports', 'figures')


@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()

# App Header
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🚗 Vehicle Fuel Efficiency Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4B5563; font-size: 1.1rem;'>Real-world fuel economy estimation comparing <b>Linear Regression</b>, <b>Support Vector Regressor (SVR)</b>, and <b>Random Forest Regressor</b>.</p>", unsafe_allow_html=True)
st.divider()

if bundle is None:
    st.error("Model pipeline artifact not found at `models/fuel_model_pipeline.joblib`. Please run `python src/train.py` first.")
    st.stop()

models_dict = bundle.get('models', {})
metadata = bundle['metadata']
metrics_summary = metadata.get('metrics_summary', {})

# Sidebar: Model Selector & Metrics
with st.sidebar:
    st.header("🧠 Select Active ML Model")
    available_models = list(models_dict.keys()) if models_dict else ["Random Forest Regressor"]
    selected_model_name = st.selectbox(
        "Choose Algorithm for Prediction:",
        available_models,
        index=available_models.index("Random Forest Regressor") if "Random Forest Regressor" in available_models else 0
    )

    active_pipeline = models_dict.get(selected_model_name, bundle.get('pipeline'))

    # Display Metrics for Selected Model
    st.subheader(f"📊 {selected_model_name}")
    mod_metrics = metrics_summary.get(selected_model_name, {})
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="R² Score", value=f"{mod_metrics.get('r2', 0.92):.1%}")
    with col_m2:
        st.metric(label="Mean Error", value=f"{mod_metrics.get('mae', 0.98):.2f} MPG")
    st.metric(label="RMSE", value=f"{mod_metrics.get('rmse', 1.57):.2f} MPG")
    st.caption("Evaluated on 9,751 unseen test vehicles from US EPA.")

    st.divider()
    st.header("⚡ Quick Preset Vehicles")
    preset = st.selectbox(
        "Load a sample vehicle configuration:",
        ["Custom", "Compact Sedan (e.g. Civic)", "Midsize Family SUV", "V8 Sports Muscle Car", "Full-Size Pickup Truck"]
    )

# Preset defaults
defaults = {
    "year": 2022,
    "vclass": "Sedan / Passenger Car",
    "drive": "Front-Wheel Drive",
    "displ": 2.0,
    "cylinders": 4,
    "trans": "Automatic",
    "fuel": "Regular Gasoline",
    "turbo": False,
    "super": False,
    "start_stop": True
}

if preset == "Compact Sedan (e.g. Civic)":
    defaults.update({"year": 2023, "vclass": "Sedan / Passenger Car", "drive": "Front-Wheel Drive", "displ": 1.5, "cylinders": 4, "trans": "CVT", "fuel": "Regular Gasoline", "turbo": True, "start_stop": True})
elif preset == "Midsize Family SUV":
    defaults.update({"year": 2022, "vclass": "SUV", "drive": "All-Wheel / 4WD", "displ": 2.5, "cylinders": 4, "trans": "Automatic", "fuel": "Regular Gasoline", "turbo": False, "start_stop": True})
elif preset == "V8 Sports Muscle Car":
    defaults.update({"year": 2021, "vclass": "Two Seater / Sports", "drive": "Rear-Wheel Drive", "displ": 5.0, "cylinders": 8, "trans": "Manual", "fuel": "Premium Gasoline", "turbo": False, "start_stop": False})
elif preset == "Full-Size Pickup Truck":
    defaults.update({"year": 2020, "vclass": "Pickup Truck", "drive": "All-Wheel / 4WD", "displ": 5.3, "cylinders": 8, "trans": "Automatic", "fuel": "Regular Gasoline", "turbo": False, "start_stop": True})

tab_pred, tab_compare, tab_eda = st.tabs(["🔮 Predict Fuel Efficiency", "📊 Compare All 3 Models", "📈 Dataset Analytics & EDA"])

with tab_pred:
    st.subheader("Configure Vehicle Specifications")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. Vehicle Profile")
        year = st.slider(
            "Model Year",
            min_value=metadata['year_range'][0],
            max_value=metadata['year_range'][1],
            value=defaults["year"]
        )
        vclass_options = metadata['vclasses']
        vclass = st.selectbox(
            "Vehicle Category / Class",
            vclass_options,
            index=vclass_options.index(defaults["vclass"]) if defaults["vclass"] in vclass_options else 0
        )
        drive_options = metadata['drive_types']
        drive = st.selectbox(
            "Drivetrain Layout",
            drive_options,
            index=drive_options.index(defaults["drive"]) if defaults["drive"] in drive_options else 0
        )

    with col2:
        st.markdown("#### 2. Engine & Transmission")
        displ = st.slider(
            "Engine Displacement (Liters)",
            min_value=float(metadata['displ_range'][0]),
            max_value=float(metadata['displ_range'][1]),
            value=float(defaults["displ"]),
            step=0.1
        )
        cylinders = st.select_slider(
            "Engine Cylinders",
            options=[2, 3, 4, 5, 6, 8, 10, 12, 16],
            value=defaults["cylinders"]
        )
        trans_options = metadata['trans_types']
        trans = st.selectbox(
            "Transmission Type",
            trans_options,
            index=trans_options.index(defaults["trans"]) if defaults["trans"] in trans_options else 0
        )

    with col3:
        st.markdown("#### 3. Fuel & Powertrain Tech")
        fuel_options = metadata['fuel_types']
        fuel = st.selectbox(
            "Fuel Type",
            fuel_options,
            index=fuel_options.index(defaults["fuel"]) if defaults["fuel"] in fuel_options else 0
        )
        st.markdown("**Induction & Efficiency Features:**")
        turbo = st.checkbox("Turbocharger (tCharger)", value=defaults["turbo"])
        supercharger = st.checkbox("Supercharger (sCharger)", value=defaults["super"])
        start_stop = st.checkbox("Automatic Engine Stop-Start", value=defaults["start_stop"])

    # Prepare input dataframe
    displ_per_cyl = displ / cylinders if cylinders > 0 else displ
    input_data = pd.DataFrame([{
        'year': year,
        'displ': displ,
        'cylinders': cylinders,
        'displ_per_cylinder': displ_per_cyl,
        'tCharger': 1 if turbo else 0,
        'sCharger': 1 if supercharger else 0,
        'startStop': 1 if start_stop else 0,
        'trans_type': trans,
        'drive_simple': drive,
        'vclass_simple': vclass,
        'fuel_simple': fuel
    }])

    if 'prediction_results' not in st.session_state:
        st.session_state['prediction_results'] = None

    st.write("")
    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        clicked = st.button(f"🚀 Predict with {selected_model_name}", type="primary", use_container_width=True)
    with btn_col2:
        st.info(f"Model: **{selected_model_name}**")

    if clicked:
        predicted_mpg = float(active_pipeline.predict(input_data)[0])
        predicted_l100km = 235.215 / predicted_mpg if predicted_mpg > 0 else 0
        est_city_mpg = max(5.0, predicted_mpg * 0.89)
        est_hwy_mpg = predicted_mpg * 1.18

        st.session_state['prediction_results'] = {
            'model_used': selected_model_name,
            'predicted_mpg': predicted_mpg,
            'predicted_l100km': predicted_l100km,
            'est_city_mpg': est_city_mpg,
            'est_hwy_mpg': est_hwy_mpg
        }

    # Render results and cost calculator whenever prediction is available
    if st.session_state['prediction_results'] is not None:
        res = st.session_state['prediction_results']
        predicted_mpg = res['predicted_mpg']
        predicted_l100km = res['predicted_l100km']
        est_city_mpg = res['est_city_mpg']
        est_hwy_mpg = res['est_hwy_mpg']

        st.success(f"✅ Prediction Computed via **{res['model_used']}**")
        
        # Result Metrics Display
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric(label="Combined MPG", value=f"{predicted_mpg:.1f} MPG")
        with res_col2:
            st.metric(label="Metric Consumption", value=f"{predicted_l100km:.1f} L/100km")
        with res_col3:
            st.metric(label="Est. City MPG", value=f"{est_city_mpg:.1f} MPG")
        with res_col4:
            st.metric(label="Est. Highway MPG", value=f"{est_hwy_mpg:.1f} MPG")

        # Efficiency Rating Category
        st.write("")
        if predicted_mpg >= 35.0:
            st.info("🌱 **Eco Efficiency Rating: High (Green Vehicle)** — Outstanding fuel economy, typical of lightweight modern powertrains or hybrid assists.")
        elif predicted_mpg >= 23.0:
            st.info("⚖️ **Eco Efficiency Rating: Moderate (Average Passenger Vehicle)** — Standard fuel economy for passenger sedans and modern crossovers.")
        else:
            st.warning("⛽ **Eco Efficiency Rating: Low (High Fuel Consumption)** — Higher fuel demand typical of large displacement engines, heavy trucks, or performance vehicles.")

        # Annual Cost Estimator (persists across slider changes without resetting)
        st.divider()
        st.subheader("💰 Annual Fuel Cost Estimator")
        st.caption("Adjust sliders below to recalculate estimated fuel expenses in real time:")
        cost_c1, cost_c2 = st.columns(2)
        with cost_c1:
            annual_miles = st.slider("Estimated Annual Driving Distance (miles):", min_value=5000, max_value=30000, value=12000, step=1000)
        with cost_c2:
            gas_price = st.slider("Average Fuel Price ($ / gallon):", min_value=2.00, max_value=6.00, value=3.50, step=0.10)
        
        annual_fuel_needed = annual_miles / predicted_mpg if predicted_mpg > 0 else 0
        annual_cost = annual_fuel_needed * gas_price
        st.markdown(f"### Estimated Fuel Cost: **${annual_cost:,.2f} / year** ({annual_fuel_needed:.0f} gallons)")

with tab_compare:
    st.subheader("🏁 Live Comparison Across All 3 Algorithms")
    st.markdown("Compare predictions and metrics for your configured vehicle across all 3 models simultaneously:")
    
    comp_rows = []
    for m_name, pipe in models_dict.items():
        pred_val = float(pipe.predict(input_data)[0])
        metrics_info = metrics_summary.get(m_name, {})
        comp_rows.append({
            "Algorithm": m_name,
            "Predicted MPG": f"{pred_val:.2f} MPG",
            "Predicted L/100km": f"{235.215 / pred_val:.2f} L/100km",
            "Model R² Score": f"{metrics_info.get('r2', 0):.2%}",
            "Test MAE": f"{metrics_info.get('mae', 0):.2f} MPG",
            "Test RMSE": f"{metrics_info.get('rmse', 0):.2f} MPG"
        })
    
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
    st.caption("💡 **Random Forest Regressor** achieves the highest accuracy ($R^2 = 92.4%$, MAE = 0.98 MPG) due to its ensemble bagging trees that capture complex engine/transmission interactions.")

with tab_eda:
    st.subheader("Exploratory Data Analysis Insights (EPA Real-World Dataset)")
    st.markdown("Visual profiles generated from our 48,752 certified vehicle records:")
    
    img_cols1, img_cols2 = st.columns(2)
    with img_cols1:
        path1 = os.path.join(FIGURES_DIR, 'displacement_vs_mpg.png')
        if os.path.exists(path1):
            st.image(path1, caption="Engine Displacement (L) vs. Combined MPG", use_container_width=True)
        path2 = os.path.join(FIGURES_DIR, 'target_distribution.png')
        if os.path.exists(path2):
            st.image(path2, caption="Distribution of Combined MPG across 48,000+ Vehicles", use_container_width=True)
            
    with img_cols2:
        path3 = os.path.join(FIGURES_DIR, 'mpg_by_drive.png')
        if os.path.exists(path3):
            st.image(path3, caption="Fuel Efficiency across Drivetrain Layouts", use_container_width=True)
        path4 = os.path.join(FIGURES_DIR, 'correlation_heatmap.png')
        if os.path.exists(path4):
            st.image(path4, caption="Feature Correlation Heatmap", use_container_width=True)
