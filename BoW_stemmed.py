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

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


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

# Calculate the total number of sentences in each dataset
total_positive_sentences = len(positive_data)
total_neutral_sentences = len(neutral_data)  # Get the number of rows (sentences)
total_negative_sentences = len(negative_data)  # Get the number of rows (sentences)

# Print the total number of sentences in each dataset
print(f"Total sentences in positive polarity dataset: {total_positive_sentences}")
print(f"Total sentences in neutral polarity dataset: {total_neutral_sentences}")
print(f"Total sentences in negative polarity dataset: {total_negative_sentences}")


# Concatenate the DataFrames into a single DataFrame
merged_dataset = pd.concat([positive_df, neutral_df, negative_df], ignore_index=True)
total_dataset = len(merged_dataset)
print(f"Total sentences in the merged dataset: {total_dataset}")

#First 5 rows of the merged data:
print(merged_dataset.tail())


# Read and process data into a DataFrame
data = []
for sentence in merged_dataset['sentence']:
    cleaned_sentence = remove_stopwords(sentence, stopwords)
    cleaned_sentence = stem_words(cleaned_sentence)
    polarity = calculate_polarity(cleaned_sentence, positive_words, negative_words)
    data.append({'sentence': sentence, 'cleaned_sentence': cleaned_sentence, 'polarity': polarity})

processed_df = pd.DataFrame(data)

# Calculate metrics using ground truth from the merged DataFrame
accuracy = accuracy_score(processed_df['polarity'], merged_dataset['polarity'])
precision = precision_score(processed_df['polarity'], merged_dataset['polarity'], average='weighted')
recall = recall_score(processed_df['polarity'], merged_dataset['polarity'], average='weighted')
f1 = f1_score(processed_df['polarity'], merged_dataset['polarity'], average='weighted')

print("********************Polarity Lexicon Results************************************************")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(processed_df['polarity'], merged_dataset['polarity'])
print("Confusion Matrix:")
print(conf_matrix)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('bow_stemmed_confusion_matrix.png')  # Save the figure as an image
plt.show()
