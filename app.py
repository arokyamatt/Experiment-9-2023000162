import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("and_gate_model.joblib")

weights = model["weights"]
bias = model["bias"]

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Streamlit UI
st.title("AND Gate using Single-Layer Neural Network")
st.write("This app simulates an AND gate trained using a perceptron-like neural network.")

# Input fields
x1 = st.number_input("Enter Input x1 (0 or 1):", min_value=0, max_value=1, step=1)
x2 = st.number_input("Enter Input x2 (0 or 1):", min_value=0, max_value=1, step=1)

if st.button("Predict"):
    # Convert to numpy array
    input_data = np.array([x1, x2])
    result = sigmoid(np.dot(input_data, weights) + bias)
    output = int(round(result[0]))
    
    st.write(f"### Predicted Output: {output}")
