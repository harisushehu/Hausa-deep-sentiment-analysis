# Sentiment Analysis in Low-Resource Languages: A Case Study on Hausa Texts

## Overview

This repository contains the code for the paper "Harisu Abdullahi Shehu, Kaloma Usman Majikumna, Aminu Bashir Suleiman, Stephen Luka, Md. Haidar Sharif, Rabie A. Ramadan, and Huseyin Kusetogullari, Unveiling Sentiments: A Deep Dive into Sentiment Analysis for Low-Resource Languages – A Case Study on Hausa Texts", which is currently under review.

## Description

## Datasets

The datasets used in this research can be accessed in the following folders:

- Translated Words: 'sentence_polarity'
- Stop Words: 'stopwords'
- Sentences: 'opinion_lexicon'

### Models
Pre-trained models can be accessed in 'models' folder.

### Bag-of-Words Approach
- **BoW.py:** Investigates the sentiment of Hausa using the bag-of-words approach.

### Stemming Method
- **BoW_stemmed.py:** Investigates sentiment analysis of Hausa texts with three categories (positive, negative, and neutral) using a stemming method proposed in our paper.
- **BoW_stemmed_binary.py:** Investigates sentiment analysis of Hausa texts with two categories (positive and negative) using the same proposed stemming method.

### Deep Learning Models
- **CNN.py:** Implements sentiment analysis using Convolutional Neural Networks (CNN).
- **HAN.py:** Implements sentiment analysis using Hierarchical Attention Network (HAN).
- **RNN.py:** Implements sentiment analysis using Recurrent Neural Networks (RNN).

- ### Comparison with Other Studies
- **comparison_other_studies.py:** Implements various machine learning algorithms, including Random Forest, Logistic Regression, K-nearest neighbors, Support Vector Machines, and Naive Bayes. These algorithms have been proposed in other studies and are utilized in this file to compare their performance with the methods employed in the current study.


### Statistical Testing
- **ANOVA.py:** Conducts a statistical test to investigate the effectiveness of the proposed methods using Analysis of Variance (ANOVA) and Tukey's test for post hoc analysis.

### Plotting
- **plot.py:** Generates bar plots illustrating accuracy, precision, recall, and F1-scores for each of the proposed methods.
- **word_cloud.py:** Generates word clouds of the positive, negative, and neutral datasets.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/harisushehu/Hausa-deep-sentiment-analysis.git

cd Hausa-deep-sentiment-analysis
```

Run the desired scripts based on your experimentation needs. For instance, run CNN.py to obtain the results of sentiment analysis using the CNN model (see below).

```bash
python CNN.py
```

Make sure to install any required dependencies in the requirements.txt


## Citation

If you use this code or datasets in your research, please cite the following paper:

```bash
@article{shehu2024unveiling,
  title={Unveiling Sentiments: A Deep Dive into Sentiment Analysis for Low-Resource Languages – A Case Study on Hausa Texts},
  author={Shehu, Harisu Abdullahi and Majikumna, Kaloma Usman and Suleiman, Aminu Bashir and Luka, Stephen and Sharif, Md. Haidar and Ramadan, Rabie A. and Kusetogullari, Huseyin},
  journal={IEEE Access},
  volume={12},
  pages={98900-98916},
  doi={\href{https://doi.org/10.1109/ACCESS.2024.3427416}{10.1109/ACCESS.2024.3427416}},
  year={2024},
  publisher={IEEE}
}
```

## Note

Contributions to the repository are welcome, and any questions can be sent to harisushehu@ecs.vuw.ac.nz.

We appreciate your interest and hope that this code proves valuable in your research endeavors.

Best regards,

The Authors



