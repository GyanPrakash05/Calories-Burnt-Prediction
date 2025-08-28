import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('calories.csv')
    return df

# Train models
@st.cache_data
def train_models(X_train, Y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'XGBoost': XGBRegressor(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'Ridge': Ridge()
    }
    for name, model in models.items():
        model.fit(X_train, Y_train)
    return models

# Streamlit App
st.set_page_config(page_title="Calories Burn Prediction", layout="wide")
st.title("🔥 Calories Burn Prediction App using Machine Learning")

# Load and preprocess data
df = load_data()

# ✅ FIXED: Encode 'Gender' early
df.replace({'male': 0, 'female': 1}, inplace=True)

# Show raw data
if st.checkbox("Show Raw Dataset"):
    st.dataframe(df.head())

# Basic Info
st.subheader("📊 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("Shape of dataset:", df.shape)
    st.write("Data Types:")
    st.write(df.dtypes)
with col2:
    st.write("Statistical Summary:")
    st.write(df.describe())

# EDA Section
st.subheader("🔍 Exploratory Data Analysis")

if st.checkbox("Show Scatterplots"):
    features = ['Age', 'Height', 'Weight', 'Duration']
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    for i, col in enumerate(features):
        sns.scatterplot(x=col, y='Calories', data=df.sample(1000), ax=axs[i//2, i%2])
    st.pyplot(fig)

if st.checkbox("Show Distributions of Float Columns"):
    float_cols = df.select_dtypes(include='float').columns
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for i, col in enumerate(float_cols):
        sns.histplot(df[col], kde=True, ax=axs[i//3, i%3])
    st.pyplot(fig)

if st.checkbox("Show Correlation Heatmap"):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr() > 0.9, annot=True, cbar=False)
    st.pyplot(fig)

# ======================
# ML Pipeline
# ======================

# Drop unwanted columns
df.drop(['Weight', 'Duration'], axis=1, inplace=True)

# Features and Target
features = df.drop(['User_ID', 'Calories'], axis=1)
target = df['Calories'].values

# Split
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train
models = train_models(X_train, Y_train)

# Evaluation
st.subheader("📈 Model Performance")
for name, model in models.items():
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    st.write(f"**{name}**")
    st.write("Training MAE:", round(mae(Y_train, train_preds), 2))
    st.write("Validation MAE:", round(mae(Y_val, val_preds), 2))
    st.markdown("---")

# Prediction
st.subheader("🧮 Predict Calories Burnt")

input_age = st.slider("Age", 10, 80, 25)
input_height = st.slider("Height (cm)", 100, 220, 170)
input_gender = st.selectbox("Gender", ['male', 'female'])
input_heart_rate = st.slider("Heart Rate", 60, 180, 90)
input_body_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

# Encode gender
input_gender = 0 if input_gender == 'male' else 1

# Combine inputs
input_data = np.array([[input_age, input_height, input_gender, input_heart_rate, input_body_temp]])
input_scaled = scaler.transform(input_data)

# Choose model
model_choice = st.selectbox("Choose Model for Prediction", list(models.keys()))

if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    st.success(f"🔥 Predicted Calories Burned: **{round(prediction, 2)} kcal**")

st.markdown("**Note**: This is a basic ML demo. Predictions may not be medically accurate.")
