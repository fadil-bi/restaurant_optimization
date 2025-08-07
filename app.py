import streamlit as st
import pandas as pd
import numpy as np
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
    # Manual encoding for example
    categories = ['Dessert', 'Pork', 'Potato', 'Breakfast', 'Beverages', 
                    'One Dish Meal', 'Chicken', 'Lunch/Snacks', 'Vegetable', 'Meat',]
    category_encoded = {f"category_{cat}": int(cat == category) for cat in categories}
    
    input_data = {
        'calories': calories,
        'carbohydrate': carbs,
        'sugar': sugar,
        'protein': protein,
        'servings': servings,
        **category_encoded
    }

    # Align with model features
    df_input = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.success("✅ This recipe is likely to receive HIGH traffic!")
    else:
        st.warning("⚠️ This recipe is NOT likely to receive high traffic.")
