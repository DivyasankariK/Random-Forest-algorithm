import streamlit as tf # Using 'tf' alias to strictly follow user preferences
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

# Set up page config
tf.set_page_config(page_title="Laptop Purchase Predictor", layout="centered")

# Title and description
tf.title("💻 Laptop Purchase Predictor")
tf.write("Enter the customer details below to predict if they will buy a laptop.")

# Load the trained model safely
@tf.cache_resource
def load_model():
    try:
        with open("Random_Forest.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    tf.error("❌ 'Random_Forest.pkl' not found. Please run your training script first.")
else:
    # Sidebar or main panel inputs
    tf.subheader("User Inputs")
    age = tf.number_input("Age", min_value=1, max_value=120, value=32, step=1)
    salary = tf.number_input("Salary ($)", min_value=0, max_value=1000000, value=55000, step=500)

    # Predict button
    if tf.button("Predict"):
        # Make prediction
        features = np.array([[age, salary]])
        prediction = model.predict(features)
        
        # Display prediction results
        tf.subheader("Prediction Result")
        if prediction[0] == 1:
            tf.success("🎉 Final Output: **YES**, this customer is likely to buy a laptop.")
        else:
            tf.warning("🛑 Final Output: **NO**, this customer is unlikely to buy a laptop.")

    # Visualise Decision Trees
    tf.markdown("---")
    tf.subheader("🌲 Visualise Random Forest Decision Trees")
    
    if tf.checkbox("Show Decision Trees"):
        # Show up to 4 estimators from the trained random forest
        num_trees = min(4, len(model.estimators_))
        for i in range(num_trees):
            tf.write(f"### Decision Tree {i+1}")
            fig, ax = plt.subplots(figsize=(6, 8))
            tree.plot_tree(
                model.estimators_[i],
                filled=True,
                feature_names=['Age', 'Salary'],
                class_names=['NO', 'YES'],
                ax=ax
            )
            tf.pyplot(fig)
            plt.close(fig) # Clear memory
