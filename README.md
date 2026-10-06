# Irrigation Water Requirement Prediction

An applied Machine Learning project that classifies agricultural irrigation water requirement into multi-class levels (**Low**, **Medium**, **High**) using environmental, soil, crop, and irrigation management features. Built with Python, Scikit-Learn, and Streamlit.

---

## 1. Project Overview

Agriculture accounts for a major portion of global freshwater consumption. Inefficient irrigation practices often cause either over-irrigation (water waste and nutrient leaching) or under-irrigation (crop water stress and diminished yield). This project builds and evaluates supervised machine learning models to determine irrigation requirements based on field measurements, crop development stages, and meteorological indicators.

The trained pipeline is deployed through an interactive Streamlit web dashboard where users can input field parameters and receive instantaneous predictions along with class confidence probabilities.

---

## 2. Problem Statement

Determining optimal irrigation timing and quantity is challenging for farmers and agronomists due to complex, non-linear interactions among:
- Soil characteristics (texture, moisture, salinity, organic carbon)
- Dynamic weather patterns (precipitation, temperature, humidity, wind)
- Crop growth phases and seasonal water demand

Without decision-support tools, irrigation decisions often rely on static schedules or guesswork, leading to suboptimal water resource management.

---

## 3. Objective

- Analyze an agricultural dataset containing 10,000 observations of soil, weather, crop, and field attributes.
- Perform exploratory data analysis (EDA), data cleaning, and preprocessing.
- Train and evaluate multiple multi-class machine learning classifiers (Logistic Regression, Decision Tree, Random Forest).
- Optimize and tune a Random Forest classification pipeline using Scikit-Learn.
- Deploy the serialized model via a user-friendly Streamlit web application.

---

## 4. Key Features

- **End-to-End Pipeline**: Unified Scikit-Learn pipeline encapsulating both data preprocessing (`ColumnTransformer` with `OneHotEncoder` and `passthrough`) and model inference (`RandomForestClassifier`).
- **Multi-Class Classification**: Predicts categorical water requirements:
  - 🟢 **Low**: Ample soil moisture / low atmospheric evaporative demand; minimal or no irrigation required.
  - 🟡 **Medium**: Moderate water demand; scheduled standard irrigation cycle.
  - 🔴 **High**: Dry soil conditions / peak demand; immediate irrigation required.
- **Probability Breakdown**: Displays the model's confidence distribution across all three target classes with visual indicators.
- **Interactive Streamlit GUI**: Clean input forms categorized logically into Soil, Weather, Crop, and Irrigation sections with median defaults from the dataset.

---

## 5. Dataset Description

The dataset (`dataset/irrigation_prediction.csv`) contains **10,000 tabular records** across **20 attributes** (19 feature columns and 1 target column).

### Input Features

| Category | Feature Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Soil** | `Soil_Type` | Categorical | Soil texture class (`Clay`, `Loamy`, `Sandy`, `Silt`) |
| **Soil** | `Soil_pH` | Numerical | Measure of soil acidity/alkalinity (0–14) |
| **Soil** | `Soil_Moisture` | Numerical | Volumetric soil moisture percentage (%) |
| **Soil** | `Organic_Carbon` | Numerical | Soil organic carbon content (%) |
| **Soil** | `Electrical_Conductivity` | Numerical | Soil salinity index (dS/m) |
| **Weather** | `Temperature_C` | Numerical | Ambient air temperature (°C) |
| **Weather** | `Humidity` | Numerical | Relative atmospheric humidity (%) |
| **Weather** | `Rainfall_mm` | Numerical | Recent rainfall depth (mm) |
| **Weather** | `Sunlight_Hours` | Numerical | Daily sunshine duration (hours) |
| **Weather** | `Wind_Speed_kmh` | Numerical | Wind speed (km/h) |
| **Crop** | `Crop_Type` | Categorical | Crop cultivated (`Cotton`, `Maize`, `Rice`, `Wheat`, etc.) |
| **Crop** | `Crop_Growth_Stage` | Categorical | Growth stage (`Initial`, `Vegetative`, `Flowering`, `Maturity`) |
| **Crop** | `Season` | Categorical | Agricultural season (`Kharif`, `Rabi`, `Zaid`) |
| **Irrigation** | `Irrigation_Type` | Categorical | Application system (`Canal`, `Drip`, `Rainfed`, `Sprinkler`) |
| **Irrigation** | `Water_Source` | Categorical | Water origin (`Canal`, `Groundwater`, `Rainwater`, `Reservoir`) |
| **Irrigation** | `Field_Area_hectare` | Numerical | Cultivated plot size (hectares) |
| **Irrigation** | `Mulching_Used` | Categorical | Soil surface mulching applied (`Yes` / `No`) |
| **Irrigation** | `Previous_Irrigation_mm`| Numerical | Depth of water supplied in previous cycle (mm) |
| **Irrigation** | `Region` | Categorical | Geographic agricultural zone (`Central`, `East`, `North`, `South`, `West`) |

### Target Variable

- **`Irrigation_Need`**: Multi-class target label:
  - `Low`
  - `Medium`
  - `High`

---

## 6. Machine Learning Approach & Workflow

```
Raw Agricultural Dataset (10,000 samples)
               │
               ▼
Exploratory Data Analysis & Validation
 (Distributions, Correlations, Value Counts)
               │
               ▼
Feature Matrix (X: 19 features) & Target (y: Irrigation_Need)
               │
               ▼
Stratified Train/Test Split (80% Train: 8,000 / 20% Test: 2,000)
               │
               ▼
ColumnTransformer Preprocessing
 ├── Categorical (8 features) ➔ OneHotEncoder(handle_unknown='ignore')
 └── Numerical (11 features)   ➔ passthrough
               │
               ▼
Model Training & Evaluation
 ├── Logistic Regression
 ├── Decision Tree Classifier
 ├── Baseline Random Forest Classifier
 └── Tuned Random Forest Classifier (max_depth=15, min_samples_split=5, n_estimators=200)
               │
               ▼
Model Serialization via Joblib ➔ models/irrigation_model.pkl
               │
               ▼
Streamlit Interactive Dashboard Deployment (app.py)
```

---

## 7. Model Evaluation & Results

All models were evaluated on the held-out test split of 2,000 samples (`test_size=0.2, random_state=42`):

| Model | Test Accuracy | Observations |
| :--- | :---: | :--- |
| **Logistic Regression** | 82.60% | Linear baseline; struggled with non-linear feature interactions |
| **Decision Tree Classifier** | 99.60% | High training and test performance, but susceptible to variance/overfitting |
| **Random Forest (Default)** | 97.35% | Strong ensemble generalization |
| **Random Forest (Tuned - Final)** | **96.95%** | Constrained depth (`max_depth=15`, `min_samples_split=5`, `n_estimators=200`) for robust generalization across unseen data |

### Final Model Classification Report (Test Set)

```
              precision    recall  f1-score   support

        High       1.00      0.40      0.57        67
         Low       0.98      1.00      0.99      1173
      Medium       0.95      0.97      0.96       760

    accuracy                           0.97      2000
   macro avg       0.98      0.79      0.84      2000
weighted avg       0.97      0.97      0.97      2000
```

---

## 8. Final Deployed Model

The final artifact deployed in the application is the **Tuned Random Forest Pipeline** stored at `models/irrigation_model.pkl`. 

It bundles:
1. `preprocessor`: `ColumnTransformer` encoding the 8 categorical features via `OneHotEncoder` and passing through 11 numerical features.
2. `model`: `RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)`.

---

## 9. Technologies Used

- **Language**: Python (3.11 – 3.13 supported)
- **Machine Learning**: `scikit-learn` (1.9.0)
- **Data Manipulation**: `pandas` (2.3.3), `numpy` (2.4.2)
- **Model Serialization**: `joblib` (1.6.0)
- **Visualization**: `matplotlib` (3.10.8), `seaborn` (0.13.2)
- **Web Interface**: `streamlit` (1.54.0)

---

## 10. Project Structure

```
Irrigation-Water-Requirement-Prediction/
│
├── app.py                  # Main Streamlit web application
├── prediction.py           # Model loading, compatibility handler, and standalone test script
├── requirements.txt        # Verified project dependencies
├── README.md               # Project documentation and guide
├── .gitignore              # Git ignore rules for Python/ML/Streamlit
│
├── dataset/
│   └── irrigation_prediction.csv    # 10,000 agricultural sample dataset
│
├── models/
│   └── irrigation_model.pkl         # Serialized Scikit-Learn pipeline
│
├── notebooks/
│   └── irrigation_prediction.ipynb  # Jupyter notebook containing complete EDA & training
│
└── graphs/                          # Directory for saved visualization outputs
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 installed on your system.
- Git installed.

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/manjulasrirama/Irrigation-Water-Requirement-Prediction-ML.git
cd Irrigation-Water-Requirement-Prediction-ML
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 12. Running the Application

### Launch Streamlit App
Run the following command from the project root:
```bash
streamlit run app.py
```
Streamlit will launch locally at:
```
http://localhost:8501
```

### Test Model Loading Script Directly
To verify that the model loads and predicts without starting the web UI:
```bash
python prediction.py
```
Output:
```
Irrigation prediction model loaded successfully!
```

### Run Jupyter Notebook
To inspect or rerun exploratory data analysis and model training:
```bash
jupyter notebook notebooks/irrigation_prediction.ipynb
```

---

## 13. Limitations

- **Multi-Class vs. Volumetric**: The model predicts categorical requirement tiers (`Low`, `Medium`, `High`) rather than exact continuous volumetric amounts (e.g., liters per hectare).
- **Static Tabular Data**: Predictions are based on user-provided snapshot inputs rather than continuous real-time IoT soil sensor or live satellite weather feeds.
- **Regional Scope**: Feature categories reflect the specific regional parameters captured in the synthetic/curated study dataset.

---

## 14. Future Enhancements

- **Volumetric Regression**: Train a complementary regression model to predict precise volume (liters or mm/ha) alongside categorical tiers.
- **Automated Weather API Integration**: Fetch real-time temperature, humidity, and rainfall forecasts directly via meteorological APIs.
- **IoT & Hardware Integration**: Interface with microcontroller-based soil moisture and EC sensors (e.g., ESP32 / Raspberry Pi) for autonomous automated irrigation scheduling.
- **Mobile Friendly PWA**: Convert dashboard into a lightweight Progressive Web App accessible directly on mobile devices by field operators.

---

## 15. Authors & Credits

- **Project**: Irrigation Water Requirement Prediction
- **Academic Domain**: Data Mining & Machine Learning
