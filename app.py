from pathlib import Path
import streamlit as st
import pandas as pd
from prediction import load_model, BASE_DIR, MODEL_PATH

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Irrigation Water Requirement Prediction",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND DATASET
# --------------------------------------------------

DATA_PATH = BASE_DIR / "dataset" / "irrigation_prediction.csv"

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_data
def get_dataset():
    return pd.read_csv(DATA_PATH)

model = get_model()
df = get_dataset()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("🌾 Project Overview")
    st.markdown(
        """
        **Irrigation Water Requirement Prediction**
        
        This system uses a trained Machine Learning pipeline 
        (Random Forest Classifier with ColumnTransformer preprocessing) 
        to predict agricultural irrigation requirements based on soil, 
        meteorological, crop, and field characteristics.
        
        ---
        **Target Classes:**
        - 🟢 **Low**: Minimal or no irrigation needed
        - 🟡 **Medium**: Moderate irrigation needed
        - 🔴 **High**: Immediate irrigation required
        
        ---
        **Dataset Summary:**
        - **Records:** 10,000 samples
        - **Features:** 19 agronomic & environmental variables
        - **Model Accuracy:** ~97% (Test Set)
        """
    )

# --------------------------------------------------
# TITLE & DESCRIPTION
# --------------------------------------------------

st.title("🌱 Irrigation Water Requirement Prediction")

st.markdown(
    "Provide the soil properties, local weather conditions, crop parameters, "
    "and irrigation details below to evaluate the recommended irrigation level."
)

st.divider()

# --------------------------------------------------
# SOIL INFORMATION
# --------------------------------------------------

st.subheader("🌍 Soil Information")

col1, col2 = st.columns(2)

with col1:
    soil_type = st.selectbox(
        "Soil Type",
        sorted(df["Soil_Type"].unique()),
        help="Type/texture of the soil in the field."
    )

    soil_ph = st.number_input(
        "Soil pH",
        value=float(df["Soil_pH"].median()),
        min_value=0.0,
        max_value=14.0,
        step=0.1,
        help="Acidity or alkalinity of the soil (0 to 14 scale)."
    )

    soil_moisture = st.number_input(
        "Soil Moisture (%)",
        value=float(df["Soil_Moisture"].median()),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        help="Current volumetric soil moisture percentage."
    )

with col2:
    organic_carbon = st.number_input(
        "Organic Carbon (%)",
        value=float(df["Organic_Carbon"].median()),
        min_value=0.0,
        step=0.05,
        help="Percentage of organic carbon in soil."
    )

    electrical_conductivity = st.number_input(
        "Electrical Conductivity (dS/m)",
        value=float(df["Electrical_Conductivity"].median()),
        min_value=0.0,
        step=0.1,
        help="Measure of soil salinity."
    )

st.divider()

# --------------------------------------------------
# WEATHER INFORMATION
# --------------------------------------------------

st.subheader("🌤️ Weather Information")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        value=float(df["Temperature_C"].median()),
        step=0.5,
        help="Ambient air temperature in degrees Celsius."
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        value=float(df["Humidity"].median()),
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        help="Relative atmospheric humidity."
    )

with col3:
    rainfall = st.number_input(
        "Rainfall (mm)",
        value=float(df["Rainfall_mm"].median()),
        min_value=0.0,
        step=1.0,
        help="Recent precipitation in millimeters."
    )

col4, col5 = st.columns(2)

with col4:
    sunlight = st.number_input(
        "Sunlight Hours",
        value=float(df["Sunlight_Hours"].median()),
        min_value=0.0,
        max_value=24.0,
        step=0.25,
        help="Average daily sunshine duration."
    )

with col5:
    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        value=float(df["Wind_Speed_kmh"].median()),
        min_value=0.0,
        step=0.5,
        help="Local wind speed."
    )

st.divider()

# --------------------------------------------------
# CROP INFORMATION
# --------------------------------------------------

st.subheader("🌾 Crop Information")

col1, col2, col3 = st.columns(3)

with col1:
    crop_type = st.selectbox(
        "Crop Type",
        sorted(df["Crop_Type"].unique()),
        help="Cultivated agricultural crop."
    )

with col2:
    crop_growth_stage = st.selectbox(
        "Crop Growth Stage",
        sorted(df["Crop_Growth_Stage"].unique()),
        help="Current biological growth phase of the crop."
    )

with col3:
    season = st.selectbox(
        "Season",
        sorted(df["Season"].unique()),
        help="Current cropping season."
    )

st.divider()

# --------------------------------------------------
# IRRIGATION & FIELD INFORMATION
# --------------------------------------------------

st.subheader("💧 Irrigation & Field Information")

col1, col2 = st.columns(2)

with col1:
    irrigation_type = st.selectbox(
        "Irrigation Type",
        sorted(df["Irrigation_Type"].unique()),
        help="Method of water application."
    )

    water_source = st.selectbox(
        "Water Source",
        sorted(df["Water_Source"].unique()),
        help="Primary source of irrigation water."
    )

    field_area = st.number_input(
        "Field Area (hectare)",
        value=float(df["Field_Area_hectare"].median()),
        min_value=0.1,
        step=0.5,
        help="Cultivated land area in hectares."
    )

with col2:
    mulching_used = st.selectbox(
        "Mulching Used",
        sorted(df["Mulching_Used"].unique()),
        help="Whether soil mulching practices are applied."
    )

    previous_irrigation = st.number_input(
        "Previous Irrigation (mm)",
        value=float(df["Previous_Irrigation_mm"].median()),
        min_value=0.0,
        step=0.5,
        help="Depth of water applied in previous irrigation cycle."
    )

    region = st.selectbox(
        "Region",
        sorted(df["Region"].unique()),
        help="Geographic agricultural region."
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Irrigation Need", use_container_width=True, type="primary"):
    try:
        # Create input dataframe matching exact model feature names
        input_data = pd.DataFrame({
            "Soil_Type": [soil_type],
            "Soil_pH": [soil_ph],
            "Soil_Moisture": [soil_moisture],
            "Organic_Carbon": [organic_carbon],
            "Electrical_Conductivity": [electrical_conductivity],
            "Temperature_C": [temperature],
            "Humidity": [humidity],
            "Rainfall_mm": [rainfall],
            "Sunlight_Hours": [sunlight],
            "Wind_Speed_kmh": [wind_speed],
            "Crop_Type": [crop_type],
            "Crop_Growth_Stage": [crop_growth_stage],
            "Season": [season],
            "Irrigation_Type": [irrigation_type],
            "Water_Source": [water_source],
            "Field_Area_hectare": [field_area],
            "Mulching_Used": [mulching_used],
            "Previous_Irrigation_mm": [previous_irrigation],
            "Region": [region]
        })

        # Make prediction
        prediction = model.predict(input_data)
        result = prediction[0]

        # Display prediction outcome
        st.subheader("🌱 Prediction Result")

        if result == "Low":
            st.success("🌱 Irrigation Requirement: **LOW** (Sufficient moisture; minimal or no irrigation needed)")
        elif result == "Medium":
            st.warning("🌱 Irrigation Requirement: **MEDIUM** (Moderate moisture required; schedule routine irrigation)")
        elif result == "High":
            st.error("🌱 Irrigation Requirement: **HIGH** (Critically low moisture / high demand; immediate irrigation needed)")
        else:
            st.info(f"🌱 Irrigation Requirement: **{result}**")

        # Prediction probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            probability_df = pd.DataFrame({
                "Irrigation Level": classes,
                "Probability (%)": (probabilities * 100).round(2)
            })

            st.subheader("📊 Prediction Probabilities")

            # Progress bars
            for _, row in probability_df.iterrows():
                level = row["Irrigation Level"]
                prob = row["Probability (%)"]
                st.write(f"**{level}: {prob:.2f}%**")
                st.progress(int(min(max(prob, 0.0), 100.0)))

            # Table view
            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                "The probabilities represent the model's confidence distribution "
                "across each irrigation requirement level."
            )

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption("Machine Learning Based Irrigation Water Requirement Prediction | Academic Data Mining Project")