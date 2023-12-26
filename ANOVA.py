#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 10:01:35 2023

@author: harisushehu
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


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






