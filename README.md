# Sentiment Analysis of the Canadian Job Market from CBC News Articles in 2024

This project focuses on analyzing sentiment trends in the Canadian job market by leveraging textual data from CBC News articles. Using advanced Natural Language Processing (NLP) techniques and machine learning models, the study aims to uncover insights into public perception and sentiment shifts in 2024.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [References](#references)


---

## Introduction

Understanding the sentiment of the Canadian job market can provide valuable insights into economic trends, public perception, and policy implications. This project processes, analyzes, and classifies news articles into sentiment categories (positive, negative, and neutral) using various embedding techniques and machine learning models. The analysis explores which embedding method and model provide the most accurate results for downstream classification tasks.

---

## Dataset

### Files:
1. `data.xlsx` - Contains unlabeled articles directly scraped using Selenium.
2. `labeled_data` - Contains data labeled by GPT-4 Mini+ with manual corrections for improved accuracy.

### Preprocessing Steps:
- Removal of duplicates and irrelevant data.
- Merging `Title` and `Description` fields into `Articles`.
- Lowercasing text, punctuation removal, and stopword filtering.
- Lemmatization applied for normalization.

---

## Methodology

### Embedding Techniques:
1. **Transformer-based Embeddings**:
   - Generated using DistilBERT, providing contextual embeddings for the text.
   - Output: \( E_{\text{DistilBERT}} \in \mathbb{R}^{n \times 768} \).
2. **Static Embeddings**:
   - Created using spaCy’s `en_core_web_sm` model for fixed 96-dimensional embeddings.

### Models:
- **Machine Learning**:
  - Logistic Regression
  - Random Forest
  - Classificatin and Regression Tree
  - Naïve Bayes
  - SVM
  - XGBoost
- **Deep Learning**:
  - Long Short-Term Memory (LSTM)

### Evaluation Metrics:
- Macro and Weighted Precision, Recall, F1-score.

---


## Technologies Used

- **Programming Language**: Python 3.11.9
- **Libraries**:
  - TensorFlow
  - Torch
  - spaCy
  - scikit-learn
  - transformers
  - pandas, NumPy, NLTK, TextBlob

---

## Installation

1. Install Python version <3.12 or check the requiremtns for torch and tensorflow version support
2. Clone this repository:
   ```bash
   git clone git@git.cs.dal.ca:courses/2024-fall/nlp-course/p-15.git
   ```
3. Set up Virtual Environment in cloned repository
4. Install the requiremtns.txt using the commanad
   ```   
   pip install -r requirements.text
   ```

##  References
1.  Andrey Shtrauss[Kaggle](https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy)
2.  Rebeen Hamad[Medium](https://medium.com/@rebeen.jaff/what-is-lstm-introduction-to-long-short-term-memory-66bd3855b9ce) 





