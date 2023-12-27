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
```

Run the desired scripts based on your experimentation needs.

```bash
python BoW.py
python BoW_stemmed.py
python BoW_stemmed_binary.py
python CNN.py
python HAN.py
python RNN.py
python ANOVA.py
python plot.py
```

Make sure to install any required dependencies in the requirements.txt

## Note

This code is part of an ongoing research project, and the associated paper is currently under review. Feel free to reach out for any inquiries or collaboration opportunities.

Contributions to the repository are welcome, and any questions can be sent to harisushehu@ecs.vuw.ac.nz.

We appreciate your interest and hope that this code proves valuable in your research endeavors.

Best regards,

The Authors



