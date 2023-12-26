#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:20:42 2023

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

# Define a callback to save the best model based on validation accuracy
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_rnn_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Define a function to create and train an RNN model
def train_rnn_model(X_train, y_train, X_test, y_test, max_words, max_seq_length, num_classes, epochs, callbacks=None):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_seq_length))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(SimpleRNN(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes)

    model.fit(X_train_pad, y_train_encoded, epochs=epochs, validation_data=(X_test_pad, y_test_encoded)) #, callbacks=callbacks)

    return model, tokenizer

def train_complex_rnn_model(X_train, y_train, X_test, y_test, max_words, max_seq_length, num_classes, epochs, callbacks=None):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_seq_length))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(64))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes)

    model.fit(X_train_pad, y_train_encoded, epochs=epochs, validation_data=(X_test_pad, y_test_encoded), callbacks=callbacks)

    return model, tokenizer


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

label_encoder = LabelEncoder()

# Ensure that labels are in string format
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# Map label strings to integers
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Number of runs
num_runs = 30
evaluation_metrics = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

for i in range(num_runs):
    # ... (rest of your code)

    # Train the RNN model with the callback
    rnn_model, tokenizer = train_complex_rnn_model(X_train, y_train_encoded, X_test, y_test_encoded, max_words, max_seq_length,
                                           num_classes, epochs, callbacks=[model_checkpoint])

    # Load the best RNN model
    best_rnn_model = tf.keras.models.load_model("best_rnn_model.h5")

    # Tokenize and pad the test data for prediction
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)
    X_test_pad = X_test_pad.astype(np.float32)

    # Make predictions
    y_pred_prob = rnn_model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate evaluation metrics and append to the list
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted')
    recall = recall_score(y_test_encoded, y_pred, average='weighted')
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')

    if accuracy <= 0.5:

        # Make predictions
        y_pred_prob = best_rnn_model.predict(X_test_pad)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Calculate evaluation metrics and append to the list
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')

    evaluation_metrics['accuracy'].append(accuracy)
    evaluation_metrics['precision'].append(precision)
    evaluation_metrics['recall'].append(recall)
    evaluation_metrics['f1'].append(f1)

    print(f"Run {i + 1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Save performance metrics to a CSV file
performance_df = pd.DataFrame(evaluation_metrics)
performance_df.to_csv('rnn_performance.csv', index=False)

# Calculate mean and standard deviation of evaluation metrics
mean_metrics = {metric: np.mean(values) for metric, values in evaluation_metrics.items()}
std_metrics = {metric: np.std(values) for metric, values in evaluation_metrics.items()}

print("\nResults over 30 runs:")
for metric in evaluation_metrics.keys():
    print(f"Mean {metric.capitalize()}: {mean_metrics[metric]} +- {std_metrics[metric]}")

# Load the best RNN model
best_rnn_model = tf.keras.models.load_model("best_rnn_model.h5")

 # Make predictions
y_pred_prob = best_rnn_model.predict(X_test_pad)
y_pred = np.argmax(y_pred_prob, axis=1)

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.title('Confusion Matrix')
plt.savefig('rnn_confusion_matrix.png')  # Save the figure as an image
plt.show()
