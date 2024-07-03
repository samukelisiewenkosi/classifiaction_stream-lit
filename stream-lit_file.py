import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Load the models and label encoder
models = {}
with open('best_lr_model.pkl', 'rb') as f:
    models['Logistic Regression'] = pickle.load(f)
with open('best_rf_model.pkl', 'rb') as f:
    models['Random Forest'] = pickle.load(f)
with open('best_xgb_model.pkl', 'rb') as f:
    models['XGBoost'] = pickle.load(f)
with open('best_mnb_model.pkl', 'rb') as f:
    models['Multinomial Naive Bayes'] = pickle.load(f)
with open('best_svm_model.pkl', 'rb') as f:
    models['SVM'] = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load and preprocess the data
@st.cache(allow_output_mutation=True)
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Combine text features into a single feature
    train_df['text'] = train_df['headlines'] + " " + train_df['description'] + " " + train_df['content']
    test_df['text'] = test_df['headlines'] + " " + test_df['description'] + " " + test_df['content']

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_train_full = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])

    # Encode the labels
    y_train_full = le.transform(train_df['category'])
    y_test = le.transform(test_df['category'])

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, class_weights))

    return train_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, le, class_weights_dict

# Load data
train_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, le, class_weights_dict = load_data()

# Title and Description
st.title("News Classification App")
st.write("Enter news text to classify it into predefined categories.")

# Input Text
user_input = st.text_area("Enter news text here", "")

# Model Selection
model_choice = st.selectbox("Choose a model", list(models.keys()))

# Prediction
if st.button("Classify"):
    if user_input:
        selected_model = models[model_choice]
        
        # Vectorize the user input using the fitted vectorizer
        user_input_vectorized = vectorizer.transform([user_input])
        
        prediction = selected_model.predict(user_input_vectorized)
        predicted_label = le.inverse_transform(prediction)
        
        st.write(f"Predicted category: {predicted_label[0]}")
    else:
        st.write("Please enter some text to classify.")