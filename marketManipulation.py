import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
# from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the CSV file
cnbcNews = pd.read_csv('cnbc_headlines.csv')

# Display the headlines of the dataframe
cnbcNews.isnull().sum()

# # Preprocessing function to clean text
# def preprocess_text(text):
#     # Convert text to lowercase
#     text = text.lower()
#     # Remove punctuation and special characters
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text

# Apply the preprocessing function to the headlines column
# newsData['Headlines'] = newsData['Headlines'].apply(preprocess_text)
# print(newsDataStr['Headlines'])

# # Tokenize the text
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(newsData['Headlines'])
# sequences = tokenizer.texts_to_sequences(newsData['Headlines'])

# # Pad the sequences
# max_sequence_length = max([len(seq) for seq in sequences])
# X = pad_sequences(sequences, maxlen=max_sequence_length)

# # Display the preprocessed and tokenized data
# print(X[:5])
