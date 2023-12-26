#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:27:05 2023

@author: harisushehu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from CSV files
cnn_data = pd.read_csv('cnn_performance.csv')
rnn_data = pd.read_csv('rnn_performance.csv')
hnn_data = pd.read_csv('hnn_performance.csv')

# Combine data into a single DataFrame
combined_data = pd.concat([cnn_data, rnn_data, hnn_data], axis=1, keys=['CNN', 'RNN', 'HNN'])

# Set up the boxplot
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
metrics = ['accuracy', 'precision', 'recall', 'f1']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=combined_data.xs(metric, axis=1, level=1))
    plt.title(f'{metric.capitalize()} Comparison')
    plt.ylabel(metric.capitalize())

plt.tight_layout()
plt.savefig("performance.png")
plt.show()
