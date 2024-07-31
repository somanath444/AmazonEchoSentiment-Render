import numpy as np
import pickle
import streamlit as st
import os
from time import sleep


# Defining the paths to the model, count vectorizer, and scalar files
model_path = os.path.join(os.path.dirname(__file__), 'model_rf.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'countVectorizer.pkl')
scalar_path = os.path.join(os.path.dirname(__file__), 'scalar.pkl')

# Loading the saved model, scalar, count vectorizer
load_model = pickle.load(open(model_path, 'rb'))
load_count_vector = pickle.load(open(vectorizer_path, 'rb'))
load_scalar = pickle.load(open(scalar_path, 'rb'))


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

