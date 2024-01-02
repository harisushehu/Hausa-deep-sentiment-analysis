#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:39:16 2024

@author: harisushehu
"""

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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

# Function to generate and save word cloud
def generate_and_save_word_cloud(df, title, filename):
    text = ' '.join(df['sentence'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')  # Save the word cloud as an image
    plt.show()

# Generate and save word clouds for each dataset
generate_and_save_word_cloud(positive_df, 'Positive Word Cloud', 'positive_word_cloud.png')
generate_and_save_word_cloud(neutral_df, 'Neutral Word Cloud', 'neutral_word_cloud.png')
generate_and_save_word_cloud(negative_df, 'Negative Word Cloud', 'negative_word_cloud.png')
