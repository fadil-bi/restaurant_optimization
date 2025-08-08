import streamlit as st
import pandas as pd
import numpy as np
from model_utils import ThresholdClassifier
import joblib 

# Load the saved model
model = joblib.load('final_model_threshold_035.pkl')

st.set_page_config(page_title="Recipe Traffic Predictor", layout="centered")
st.title("📈 Recipe Traffic Prediction")
st.markdown('''
This app predicts whether a recipe will likely become **high traffic** based on nutritional and categorical information. 
Designed for restaurant use to optimize menu items.
''')

# Input form
with st.form("prediction_form"):
    calories = st.number_input("Total Calories", min_value=0.0, step=1.0)
    carbohydrate = st.number_input("Carbohydrates (g)", min_value=0.0, step=1.0)
    sugar = st.number_input("Sugar (g)", min_value=0.0, step=1.0)
    protein = st.number_input("Protein (g)", min_value=0.0, step=1.0)
    servings = st.number_input("Servings", min_value=1, step=1)
    
    category = st.selectbox("Recipe Category", [
        'Dessert', 'Pork', 'Potato', 'Breakfast', 'Beverages', 
        'One Dish Meal', 'Chicken', 'Lunch/Snacks', 'Vegetable', 'Meat',
    ])
    
    submit = st.form_submit_button("Predict")

# Preprocess the input
if submit:
    # Calculate calories per portion
    calorie_portion = calories / servings if servings > 0 else 0.0

    input_data = {
        'calories': calories,
        'carbohydrate': carbohydrate,
        'sugar': sugar,
        'protein': protein,
        'servings': servings,
        'calorie_portion': calorie_portion,
        'category': category  # NOTE: pass raw, let model handle encoding
    }

    df_input = pd.DataFrame([input_data])
    
    # Prediction
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.success("✅ This recipe is likely to receive HIGH traffic!")
    else:
        st.warning("⚠️ This recipe is NOT likely to receive high traffic.")
