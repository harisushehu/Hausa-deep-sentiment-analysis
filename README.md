# Deep Learning for Sentiment Analysis in Low-Resource Languages

## Overview

This repository contains the code for the paper titled "Deep Learning for Sentiment Analysis in Low-Resource" authored by Harisu Abdullahi Shehu, Kaloma Usman Majikumna, Aminu Bashir Suleiman, Stephen Luka, Md. Haidar Sharif, Rabie A. Ramadan, and Huseyin Kusetogullari. As of now, the paper is under review.

## Description

### Bag-of-Words Approach
- **BoW.py:** Investigates the sentiment of Hausa using the bag-of-words approach.

### Stemming Method
- **BoW_stemmed.py:** Investigates sentiment analysis of Hausa texts with three categories (positive, negative, and neutral) using a stemming method proposed in our paper.
- **BoW_stemmed_binary.py:** Investigates sentiment analysis of Hausa texts with two categories (positive and negative) using the same proposed stemming method.

### Deep Learning Models
- **CNN.py:** Implements sentiment analysis using Convolutional Neural Networks (CNN).
- **HAN.py:** Implements sentiment analysis using Hierarchical Attention Network (HAN).
- **RNN.py:** Implements sentiment analysis using Recurrent Neural Networks (RNN).

### Statistical Testing
- **ANOVA.py:** Conducts a statistical test to investigate the effectiveness of the proposed methods using Analysis of Variance (ANOVA) and Turkey's test for post hoc analysis.

### Plotting
- **plot.py:** Generates bar plots illustrating accuracy, precision, recall, and F1-scores for each of the proposed methods.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository

