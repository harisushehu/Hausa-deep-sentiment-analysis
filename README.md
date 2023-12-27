# Deep Learning for Sentiment Analysis in Low-Resource Languages: A Case Study on Hausa Texts

## Overview

This repository contains the code for the paper "Harisu Abdullahi Shehu, Kaloma Usman Majikumna, Aminu Bashir Suleiman, Stephen Luka, Md. Haidar Sharif, Rabie A. Ramadan, and Huseyin Kusetogullari, Deep Learning for Sentiment Analysis in Low-Resource Languages: A Case Study on Hausa Texts", which is currently under review.

## Description

## Datasets

The datasets used in this research can be accessed in the following folders:

- Translated Words: 'sentence_polarity'
- Stop Words: 'stopwords'
- Sentences: 'opinion_lexicon'

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

## Citation

If you use this code or datasets in your research, please cite the following paper:

> Harisu Abdullahi Shehu, Kaloma Usman Majikumna, Aminu Bashir Suleiman, Stephen Luka, Md. Haidar Sharif, Rabie A. Ramadan, and Huseyin Kusetogullari, Deep Learning for Sentiment Analysis in Low-Resource Languages: A Case Study on Hausa Texts.

```bash
@article{shehu2023deep,
  title={Deep Learning for Sentiment Analysis in Low-Resource Languages: A Case Study on Hausa Texts},
  author={Shehu, Harisu Abdullahi and Majikumna, Kaloma Usman and Suleiman, Aminu Bashir and Luka, Stephen and Sharif, Md. Haidar and Ramadan, Rabie A. and Kusetogullari, Huseyin},
  year={2023},
  journal={Journal Name (Under Review)},
  note={Manuscript under review.},
}
```

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



