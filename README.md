# AI-Task02
This project focuses on sentiment analysis using the Sentiment140 dataset. It employs both rule-based and machine learning-based approaches to classify the sentiment of tweets as positive or negative. The project covers data preprocessing, text analysis, feature extraction, model training, and evaluation, providing comprehensive insights into the sentiment classification process and its effectiveness.

### Table of Contents:

1. [Dataset Information](#dataset-information)
2. [Installation](#installation)
3. [Importing Libraries & Loading Dataset](#importing-libraries-and-loading-dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Sentiment Analysis with NLTK Vader](#sentiment-analysis-with-nltk-vader)
6. [Machine Learning-Based Sentiment Analysis](#machine-learning-based-sentiment-analysis)
7. [Evaluation](#evaluation)
8. [Conclusion](#conclusion)

## Dataset Information

- **Dataset Name**: [Sentiment140](https://huggingface.co/datasets/contemmcm/sentiment140)
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets)
- **Description**: The dataset contains 1,600,000 tweets with preassigned sentiment labels (`0` for `negative` and `1` for `positive`).

## Installation

To get started, install the necessary libraries:

```bash
pip install datasets nltk tqdm scikit-learn
```

## Importing Libraries and Loading Dataset

- **Import Libraries**:
   - `Pandas` for data manipulation.
   - `NLTK` for natural language processing - sentiment analysis.
   - `Scikit-learn` for evaluation metrics.
   - `Datasets` library to load data.
   - `re` regex to preprocess data.

- **Load Dataset**:
   - Load the Sentiment140 dataset using the `datasets` library.
   - Convert the dataset to a Pandas DataFrame for easier manipulation.

## Data Preprocessing

- **Text Preprocessing**:
   - Remove special characters.
   - Tokenize text.
   - Remove stop words.
   - Lemmatize tokens.

## Sentiment Analysis with NLTK Vader

- **Initialize Sentiment Analyzer**:
   - Use NLTK's Vader sentiment analyzer to determine the sentiment of each text entry.

- **Calculate Sentiment**:
   - Define a function to get sentiment scores and classify the text as positive or negative based on the scores.

## Machine Learning-Based Sentiment Analysis

- **Prepare Data**:
   - Tokenize the text data and create a vocabulary of words.
   - Generate features for each document based on the presence of words in the vocabulary.

- **Train-Test Split**:
   - Split the dataset into training and testing sets.

- **Train Classifier**:
   - Train a Naive Bayes classifier on the training set.

## Evaluation

- **Confusion Matrix and Classification Report**:
  - Evaluate the performance of the NLTK Vader sentiment analyzer using a confusion matrix and classification report.
  
- **Classifier Accuracy**:
  - Evaluate the performance of the Naive Bayes classifier using accuracy metrics.

## Conclusion

This project demonstrates how to perform sentiment analysis using both rule-based and machine learning-based approaches. The results show the effectiveness of each method and provide insights into the sentiment of text data.
