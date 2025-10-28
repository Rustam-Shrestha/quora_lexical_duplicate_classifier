import streamlit as st
import helper
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Duplicate Question Detector", layout="centered")

# Title and description
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Duplicate Question Detector</h1>", unsafe_allow_html=True)
st.markdown("Enter two questions below to check if they are semantically similar.")

# Input fields
q1 = st.text_input('Question 1')
q2 = st.text_input('Question 2')

# Prediction logic
if st.button('Check for Duplicate'):
    if q1.strip() == "" or q2.strip() == "":
        st.warning("Please enter both questions.")
    else:
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]

        if result:
            st.markdown("<h3 style='color: green;'>Result: Duplicate</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: red;'>Result: Not Duplicate</h3>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Powered by Streamlit</p>", unsafe_allow_html=True)
