#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK data files
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Loading the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Keeping only the relevant columns
df = df[['v1', 'v2']]

# Renaming the columns for easier access
df.columns = ['label', 'message']

# Initializing stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    # Removing special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Converting to lowercase
    text = text.lower()
    # Removing stopwords and lemmatizing
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words and len(word) > 2)
    return text

# Applying text cleaning to the messages
df['cleaned_message'] = df['message'].apply(clean_text)

# TF-IDF Vectorization on the cleaned data
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['cleaned_message'])
y = df['label']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Multinomial Naive Bayes classifier
classifier = MultinomialNB(alpha=0.1)

# Training the classifier on the training data
classifier.fit(X_train, y_train)

# Predicting the labels for the validation set
y_val_pred = classifier.predict(X_val)

# Calculating accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Function to predict label for user input
def predict_label(message):
    cleaned_message = clean_text(message)
    message_vector = tfidf_vectorizer.transform([cleaned_message])
    predicted_label = classifier.predict(message_vector)[0]
    return predicted_label

# User interaction for predicting labels
print("\nEnter an SMS message to predict whether it's spam or not (or 'quit' to exit):")
while True:
    user_input = input("SMS Message: ")
    if user_input.lower() == 'quit':
        break
    else:
        label = predict_label(user_input)
        print(f"The predicted label for the given message is: {label}")
