#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:46:08 2024

@author: harisushehu
"""

import os
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.layers import Dropout

# Define functions for text preprocessing
def remove_stopwords(sentence, stopwords):
    words = sentence.lower().split()
    words = [word for word in words if word not in stopwords]
    return " ".join(words)

def tokenize(sentence):
    return word_tokenize(sentence)


with open("./sentence_polarity/positive-words-HA.txt", "r") as file:
    positive_words = file.read().splitlines()

with open("./sentence_polarity/negative-words-HA.txt", "r") as file:
    negative_words = file.read().splitlines()

with open("./stopwords/StopWords-HA.txt", "r") as file:
    stopwords = file.read().splitlines()

# Read positive dataset into a DataFrame
with open('./opinion_lexicon/polarity-positive-HA.txt', 'r') as file:
    positive_data = [{'sentence': line.strip(), 'polarity': 'Positive'} for line in file]

positive_df = pd.DataFrame(positive_data)

# Read neutral dataset into a DataFrame
with open('./opinion_lexicon/polarity-neutral-HA.txt', 'r') as file:
    neutral_data = [{'sentence': line.strip(), 'polarity': 'Neutral'} for line in file]

neutral_df = pd.DataFrame(neutral_data)

# Read negative dataset into a DataFrame
with open('./opinion_lexicon/polarity-negative-HA.txt', 'r') as file:
    negative_data = [{'sentence': line.strip(), 'polarity': 'Negative'} for line in file]

negative_df = pd.DataFrame(negative_data)

# Concatenate the DataFrames into a single DataFrame
merged_dataset = pd.concat([positive_df, neutral_df, negative_df], ignore_index=True)
total_dataset = len(merged_dataset)
print(f"Total sentences in the merged dataset: {total_dataset}")

# Read and process data into a DataFrame
data = []
for sentence, polarity in zip(merged_dataset['sentence'], merged_dataset['polarity']):
    cleaned_sentence = remove_stopwords(sentence, stopwords)
    # Strip any leading or trailing white spaces or newlines from the polarity value
    polarity = polarity.strip()
    data.append({'sentence': sentence, 'cleaned_sentence': cleaned_sentence, 'polarity': polarity})

processed_df = pd.DataFrame(data)

# Define the desired percentage of each class in the test set
test_set_percentage = 0.2  # 20% of each class

# Separate the data by polarity
positive_samples = processed_df[processed_df['polarity'] == 'Positive']
neutral_samples = processed_df[processed_df['polarity'] == 'Neutral']
negative_samples = processed_df[processed_df['polarity'] == 'Negative']

# Split each class into train and test sets
positive_train, positive_test = train_test_split(positive_samples, test_size=test_set_percentage, random_state=42)
neutral_train, neutral_test = train_test_split(neutral_samples, test_size=test_set_percentage, random_state=42)
negative_train, negative_test = train_test_split(negative_samples, test_size=test_set_percentage, random_state=42)

# Concatenate the train and test sets for each class
X_train = pd.concat([positive_train['cleaned_sentence'], neutral_train['cleaned_sentence'], negative_train['cleaned_sentence']], ignore_index=True)
y_train = pd.concat([positive_train['polarity'], neutral_train['polarity'], negative_train['polarity']], ignore_index=True)
X_test = pd.concat([positive_test['cleaned_sentence'], neutral_test['cleaned_sentence'], negative_test['cleaned_sentence']], ignore_index=True)
y_test = pd.concat([positive_test['polarity'], neutral_test['polarity'], negative_test['polarity']], ignore_index=True)

print("Number of Neutral Test Samples:", len(neutral_test))
print("Number of Positive Test Samples:", len(positive_test))
print("Number of Negative Test Samples:", len(negative_test))


# Parameters for the CNN model
max_words = 10000  # Maximum number of words to keep
max_seq_length = 100  # Maximum sequence length (pad or truncate to this length)
num_classes = 3  # Number of classes (Negative, Neutral, Positive)
epochs = 10  # Number of training epochs

tokenizer = Tokenizer(num_words=max_words)

label_encoder = LabelEncoder()

# Ensure that labels are in string format
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# Map label strings to integers
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Load pre-trained word embeddings (GloVe embeddings)
embedding_dim = 100  # You may need to adjust this based on the dimension of the GloVe embeddings you downloaded
embedding_matrix = {}
with open('./glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_matrix[word] = coefs

# Create a matrix with the words in your tokenizer and their corresponding embeddings
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index)) + 1
embedding_matrix_final = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i > max_words:
        continue
    embedding_vector = embedding_matrix.get(word)
    if embedding_vector is not None:
        embedding_matrix_final[i] = embedding_vector

# Parameters for the machine learning classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=110, max_depth=20, random_state=14),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=14),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='poly', C=1.0, random_state=14),
    'Naive Bayes': MultinomialNB(alpha=1.0)
}

# Number of runs
num_runs = 30
evaluation_metrics = {classifier: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for classifier in classifiers.keys()}

# Tokenize and pad the train data for prediction
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
X_train_pad = X_train_pad.astype(np.float32)

# Tokenize and pad the test data for prediction
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)
X_test_pad = X_test_pad.astype(np.float32)

for i in range(num_runs):
    for classifier_name, classifier in classifiers.items():

        # Train and predict using the current classifier
        classifier.fit(X_train_pad, y_train_encoded)

        y_pred = classifier.predict(X_test_pad)

        # Calculate evaluation metrics and append to the list
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')

        evaluation_metrics[classifier_name]['accuracy'].append(accuracy)
        evaluation_metrics[classifier_name]['precision'].append(precision)
        evaluation_metrics[classifier_name]['recall'].append(recall)
        evaluation_metrics[classifier_name]['f1'].append(f1)

        print(f"Run {i + 1} - {classifier_name}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save performance metrics to CSV files for each classifier
for classifier_name, metrics_dict in evaluation_metrics.items():
    performance_df = pd.DataFrame(metrics_dict)
    performance_df.to_csv(f'{classifier_name.lower()}_performance.csv', index=False)

# Calculate mean and standard deviation of evaluation metrics for each classifier
mean_metrics = {classifier_name: {metric: np.mean(values) for metric, values in metrics_dict.items()} for classifier_name, metrics_dict in evaluation_metrics.items()}
std_metrics = {classifier_name: {metric: np.std(values) for metric, values in metrics_dict.items()} for classifier_name, metrics_dict in evaluation_metrics.items()}

# Print results for each classifier
print("\nResults over 30 runs for each classifier:")
for classifier_name, metrics_dict in mean_metrics.items():
    print(f"\nClassifier: {classifier_name}")
    for metric, mean_value in metrics_dict.items():
        std_value = std_metrics[classifier_name][metric]
        print(f"Mean {metric.capitalize()}: {mean_value} +- {std_value}")

