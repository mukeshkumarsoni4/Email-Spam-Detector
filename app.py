import streamlit as st
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK objects
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)

@st.cache_data
def load_data():
    data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
    data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    data.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['label'])
    return data

# Load data
data = load_data()

# Split data into features and labels
X = data['message']
y = data['label']

# Preprocess the text
X_transformed = X.apply(transform_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X_transformed)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")