import pickle
import numpy as np
import streamlit as st
import pandas as pd
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained sentiment analysis model
model = pickle.load(open("sentiment_deploy", "rb"))

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Streamlit app title
st.set_page_config(layout="wide")
st.title("🌞 Sentiment Analysis 🌍")
st.markdown("<h2>Predict the sentiment of reviews using Trained Model.</h2>", unsafe_allow_html=True)
st.header("📂 Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
if st.button("🌟 Predict Sentiment for CSV"):
        if uploaded_file is not None:
            # Read the CSV file
            data = pd.read_csv(uploaded_file)
            
            # Drop missing values and remove 'Unknown' reviews
            data.dropna(inplace=True)
            data = data[data['Review'] != "Unknown"]
            
            # Reset index after dropping rows
            data.reset_index(drop=True, inplace=True)
            
            # Check if the CSV has a 'Review' column
            if 'Review' in data.columns:
                # Clean the review text
                data['Cleaned_Review'] = data['Review'].astype(str).apply(clean_text)
                
                # Transform the cleaned text using the TF-IDF vectorizer
                X = tfidf.transform(data['Cleaned_Review'])
                
                # Predict sentiment for each review
                predictions = model.predict(X)
                
                # Map numerical predictions to sentiment labels
                sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                data['Predicted_Sentiment'] = [sentiment_labels[pred] for pred in predictions]
                
                # Display predictions
                st.write("## Sentiment Predictions:")
                st.dataframe(data[['Review', 'Predicted_Sentiment']].rename(columns={'Review': 'Customer Review', 'Predicted_Sentiment': 'Sentiment'}))
            else:
                st.error("The CSV file must contain a 'Review' column for sentiment analysis.")
        else:
            st.error("Please upload a CSV file to proceed.")