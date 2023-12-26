#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:41:16 2023

@author: harisushehu
"""

import os
import pandas as pd
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def remove_stopwords(sentence, stopwords):
    words = sentence.lower().split()
    words = [word for word in words if word not in stopwords]
    return " ".join(words)


def tokenize(sentence):
    return word_tokenize(sentence)


def stem_words(sentence):
    # Remove 'r' and 'n' from the end of each word
    stemmed_words = [word.rstrip('rn') for word in sentence.lower().split()]
    return " ".join(stemmed_words)


def calculate_polarity(sentence, positive_words, negative_words):
    polarity = 0
    words = tokenize(sentence)

    for word in words:
        if word in positive_words:
            polarity += 1
        elif word in negative_words:
            polarity -= 1

    if polarity >= 0:  # Change from > 0 to >= 0 for positive polarity
        return "Positive"
    else:
        return "Negative"


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

# Read negative dataset into a DataFrame
with open('./opinion_lexicon/polarity-negative-HA.txt', 'r') as file:
    negative_data = [{'sentence': line.strip(), 'polarity': 'Negative'} for line in file]

negative_df = pd.DataFrame(negative_data)

test_data = pd.concat([positive_df, negative_df], ignore_index=True)

# Read and process data into a DataFrame for the test set
test_data_processed = []
for sentence in test_data['sentence']:
    cleaned_sentence = remove_stopwords(sentence, stopwords)
    cleaned_sentence = stem_words(cleaned_sentence)
    polarity = calculate_polarity(cleaned_sentence, positive_words, negative_words)
    test_data_processed.append({'sentence': sentence, 'cleaned_sentence': cleaned_sentence, 'polarity': polarity})

test_processed_df = pd.DataFrame(test_data_processed)

# Calculate metrics using ground truth from the merged DataFrame
accuracy = accuracy_score(test_processed_df['polarity'], test_data['polarity'])
precision = precision_score(test_processed_df['polarity'], test_data['polarity'], average='weighted')
recall = recall_score(test_processed_df['polarity'], test_data['polarity'], average='weighted')
f1 = f1_score(test_processed_df['polarity'], test_data['polarity'], average='weighted')

print("********************Polarity Lexicon Results************************************************")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Calculate and print the confusion matrix for the test set
conf_matrix_test = confusion_matrix(test_processed_df['polarity'], test_data['polarity'])
print("Confusion Matrix (Test Set):")
print(conf_matrix_test)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('bow_stemmed_confusion_matrix_test.png')  # Save the figure as an image
plt.show()


