import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns

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

# Function to evaluate model performance
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=le.classes_)
    return accuracy, report, y_pred

# Load data
train_df, test_df, X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, le, class_weights_dict = load_data()

# Main Page - News Classification App
def main():
    st.title("News Classification App")
    st.write("""
    Welcome to the News Classification App! This application uses machine learning models to classify news articles 
    into predefined categories based on their content. Whether you're a reader, journalist, or researcher, this tool 
    provides an efficient way to categorize news stories automatically.

    Explore different models to see how they classify news articles differently. Feel free to reach out for any questions or feedback using the 'Contact Information' section on the sidebar.

    """)

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
            
            # Model Evaluation
            st.subheader("Model Evaluation on Validation Set")
            accuracy, report, y_pred = evaluate_model(selected_model, X_val, y_val)
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write("Classification Report:")
            st.write(report)
            
            # Display confusion matrix using seaborn
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            st.pyplot()

        else:
            st.write("Please enter some text to classify.")

# Sidebar Pages
def sidebar_pages():
    st.sidebar.title("Navigation")
    pages = {
        "Introduction": introduction,
        "User Guide": user_guide,
        "Home": main,
        "Recommendations": recommendations,
        "Contact Info": contact_info
    }
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

# Additional Pages
def introduction():
    st.title("Welcome to News Classification App")
    st.write("""
    This application provides an automated way to classify news articles into specific categories using machine learning models. 
    You can choose from several models including Logistic Regression, Random Forest, XGBoost, Multinomial Naive Bayes, and Support 
    Vector Machine (SVM). Each model has been trained on a dataset of news articles to predict the most suitable category based on 
    the content provided.

    ### Why Use This App?
    - **Efficiency:** Save time by automating the categorization process.
    - **Accuracy:** Benefit from models trained on extensive datasets to make informed predictions.
    - **Versatility:** Explore different models to see how they categorize news articles differently.

    Use the sidebar navigation to explore recommendations, user guides, and contact information for support or inquiries.

    """)

def recommendations():
    st.title("Recommendations")
    st.write("""
    Recommendations for using this application effectively:
    - Enter complete news text for accurate classification.
    - Select different models to compare classification results.
    - Evaluate model performance using the provided metrics and visualizations.

    Stakeholders Recommendations:
    Editorial Team:
    - Regular Content Review: Periodically review categorized content to ensure accuracy and relevance. Provide feedback on misclassifications to help refine the models.
    - Training and Adoption: Participate in training sessions to fully leverage the features of the Streamlit application. Encourage the editorial team to adopt the new system for consistent use.
    - Content Insights: Utilize additional features such as trend analysis and sentiment analysis to gain insights into content performance and audience engagement.

    IT/Tech Support:
    - System Maintenance: Establish a regular maintenance schedule to update the application and models. Ensure the system is running smoothly and troubleshoot any technical issues promptly.
    - Scalability Planning: Plan for scalability to handle increased data volume and user load. Consider cloud deployment for enhanced scalability and reliability.
    - Integration: Work on integrating the Streamlit application with existing content management systems for seamless operations.

    Management:
    - Performance Monitoring: Regularly monitor the performance metrics of the classification models and the overall system. Make data-driven decisions based on these insights.
    - Resource Allocation: Allocate resources for continuous improvement of the models and application. Support initiatives for regular updates and advancements.
    - Stakeholder Feedback: Collect feedback from all stakeholders and incorporate it into the development roadmap to ensure the system meets their evolving needs.
    """)

def user_guide():
    st.title("User Guide")
    st.write("""
    Follow these steps to use the News Classification App:
    1. Enter news text in the text area on the main page.
    2. Choose a model from the dropdown list.
    3. Click the 'Classify' button to see the predicted category.
    4. View model evaluation metrics and confusion matrix for validation.
             
After classification, you can view model evaluation metrics such as accuracy, classification report, and confusion matrix on the validation set to assess the model's performance.
    """)

def contact_info():
    st.title("Contact Information")
    st.write("""
    For support or inquiries, please contact:
    - Email: team-mm5@exploreai.com
    - Phone: +1234567890

    Address:
    - 377 Gang Street
    - Johannesburg
    - Gauteng
    """)

# Run the app
if __name__ == "__main__":
    sidebar_pages()