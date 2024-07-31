import numpy as np
import pickle
import streamlit as st
import os
from time import sleep

#Construct paths relative to the current working directory
model_path = os.path.join('model_rf.pkl')
count_vector_path = os.path.join('countVectorizer.pkl')
scalar_path = os.path.join('scalar.pkl')

# Loading the saved model, scalar, count vectorizer
load_model = pickle.load(open('D:\\Deployments\\amazonecho_sentiment_analysis\\model_rf.pkl', 'rb'))
load_count_vector = pickle.load(open('D:\\Deployments\\amazonecho_sentiment_analysis\\countVectorizer.pkl', 'rb'))
load_scalar = pickle.load(open('D:\\Deployments\\amazonecho_sentiment_analysis\\scalar.pkl', 'rb'))

# Creating a function for prediction
def sentiment_prediction(input_data):
    input_1_vectors = load_count_vector.transform([input_data])
    input_1_dense = input_1_vectors.toarray()
    input_1_scaled = load_scalar.transform(input_1_dense)
    prediction = load_model.predict(input_1_scaled)
    sentiment = 'Positive review' if prediction[0] == 1 else 'Negative review'
    return sentiment

# Streamlit UI
def main():
    st.title('Sentiment Analysis Prediction')
    st.markdown("<h1 style='text-align: center; color: purple;'>Welcome to the Sentiment Analyzer!</h1>", unsafe_allow_html=True)
    
    input_data = st.text_area('Enter the review text: ')
    
    if st.button('Predict'):
        with st.spinner('Processing...'):
            progress_bar = st.progress(0)
            for i in range(1, 101):
                sleep(0.02)  # Faster progress bar update
                progress_bar.progress(i)
            prediction = sentiment_prediction(input_data)
            st.success(f"The Predicted Sentiment is: {prediction}")
            
            # Display a success message with an icon
            st.markdown("""
                <div style='text-align: center; color: green;'>
                    <h2>🎉 Prediction Complete!</h2>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

