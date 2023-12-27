#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:01:35 2023

@author: harisushehu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Read data from CSV files
cnn_data = pd.read_csv('cnn_performance.csv')
rnn_data = pd.read_csv('rnn_performance.csv')
hnn_data = pd.read_csv('hnn_performance.csv')

# Extract accuracy columns from each DataFrame
cnn_accuracy = cnn_data['accuracy']
rnn_accuracy = rnn_data['accuracy']
hnn_accuracy = hnn_data['accuracy']

# Combine data into a single DataFrame
combined_data = pd.concat([cnn_accuracy, rnn_accuracy, hnn_accuracy], axis=1, keys=['CNN', 'RNN', 'HNN'])

# Set up the boxplot
plt.figure(figsize=(15, 10))
sns.set_theme(style="whitegrid")
metrics = ['accuracy']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=combined_data)
    plt.title(f'{metric.capitalize()} Comparison')
    plt.ylabel(metric.capitalize())

# ANOVA analysis
f_statistic, p_value_anova = f_oneway(cnn_accuracy, rnn_accuracy, hnn_accuracy)
print(f'ANOVA results: F-statistic={f_statistic}, p-value={p_value_anova}')

# Combine data into a single DataFrame
flattened_data = pd.concat([cnn_accuracy, rnn_accuracy, hnn_accuracy]).values.flatten()
group_labels = ['CNN'] * len(cnn_accuracy) + ['RNN'] * len(rnn_accuracy) + ['HNN'] * len(hnn_accuracy)

# Perform Tukey's test
tukey_results = pairwise_tukeyhsd(flattened_data, group_labels)


# Plot significance results in a table
tukey_summary = tukey_results.summary()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # Turn off the axis

# Create a table from the tukey summary data
table_data = []
for row in tukey_summary.data:
    table_data.append([row[0], row[1], row[2], row[3]])

table_headers = ['Group 1', 'Group 2', 'Mean Diff', 'P-Value']
ax.table(cellText=table_data, colLabels=table_headers, loc='center')

plt.savefig("performance_with_significance.png")
plt.show()


#************************************ ANOVA detailed result ************************************************#

# Read data from CSV files
cnn_data = pd.read_csv('cnn_performance.csv')
rnn_data = pd.read_csv('rnn_performance.csv')
hnn_data = pd.read_csv('hnn_performance.csv')

# Add 'Method' column to each DataFrame
cnn_data['Method'] = 'CNN'
rnn_data['Method'] = 'RNN'
hnn_data['Method'] = 'HAN'

# Merge DataFrames into a single DataFrame
merged_df = pd.concat([cnn_data, rnn_data, hnn_data], ignore_index=True)

Performance = merged_df['accuracy']
Performance = Performance * 100

# Fit the OLS model
formula = 'accuracy ~ C(Method)'
results = ols(formula, data=merged_df).fit()

# ANOVA table
aov_table = sm.stats.anova_lm(results, typ=1)
aov_table

# Function to calculate mean squared, eta squared, and omega squared
def anova_table(aov):
    aov['mean_sq'] = aov['sum_sq'] / aov['df']
    aov['eta_sq'] = aov['sum_sq'] / (aov['sum_sq'] + aov['df'][1] * aov['mean_sq'][0])
    aov['omega_sq'] = (aov['sum_sq'] - (aov['df'][0] * aov['mean_sq'][1])) / (aov['sum_sq'] + aov['mean_sq'][1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

# Calculate ANOVA table with mean squared, eta squared, and omega squared
anova_table = anova_table(aov_table)
print(anova_table)

