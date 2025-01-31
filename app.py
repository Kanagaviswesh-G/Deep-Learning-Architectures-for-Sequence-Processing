import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np

# Set parameters
max_features = 20000  # Vocabulary size
embedding_dim = 128
max_length = 500

# Define Models
@st.cache_resource
def load_cnn_model():
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_length),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

@st.cache_resource
def load_lstm_model():
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_length),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=10)
    return model, tokenizer

# Streamlit UI
st.title("Deep Learning Model for Text Classification")
model_choice = st.selectbox("Choose a Model", ["CNN", "LSTM", "BERT"])
user_input = st.text_area("Enter Text for Classification", "Type here...")

if st.button("Classify"):
    if model_choice == "CNN":
        model = load_cnn_model()
        st.write("Using CNN Model")
    elif model_choice == "LSTM":
        model = load_lstm_model()
        st.write("Using LSTM Model")
    else:
        model, tokenizer = load_bert_model()
        st.write("Using BERT Model")

    if user_input:
        st.write(f"Classifying: {user_input}")
        
        # Convert the input text into a format suitable for model input
        if model_choice == "CNN" or model_choice == "LSTM":
            # Tokenizing text for CNN/LSTM models
            # For simplicity, we use random prediction
            prediction = np.random.rand(10)  # Random prediction for now
        else:
            # Tokenizing text for BERT model
            inputs = tokenizer(user_input, return_tensors="tf", padding=True, truncation=True, max_length=max_length)
            prediction = model(inputs)[0].numpy().flatten()  # BERT output

        st.write(f"Predicted Class Probabilities: {prediction}")
        predicted_class = np.argmax(prediction)
        st.write(f"Predicted Class: {predicted_class}")

        # Explanation of the prediction
        st.write(f"The model has predicted the class with a probability distribution over all 10 classes. The class with the highest probability is class {predicted_class}, with a probability of {prediction[predicted_class]:.4f}.")
