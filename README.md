# Fake News Prediction System

This project implements a Fake News Prediction System using Machine Learning with Python. The Logistic Regression model is utilized for prediction. Below is an overview of the project workflow and the libraries used.

## Project Overview

The goal of this project is to predict the likelihood of news articles being fake using machine learning techniques. The system is built with the following steps:

1. **Data Collection**: Importing necessary libraries and the dataset.
2. **Data Preprocessing**: Cleaning and preparing the data for analysis.
3. **Text Preprocessing**: Applying natural language processing techniques to process the text data.
4. **Feature Extraction**: Transforming text data into numerical features.
5. **Model Training**: Training the Logistic Regression model on the processed data.
6. **Model Evaluation**: Assessing the performance of the model using accuracy scores.

## Libraries Used

- **numpy**: This library is used for numerical operations, providing support for arrays and matrices.
- **pandas**: A powerful data manipulation tool that helps with data analysis and cleaning.
- **re**: This library assists in finding and working with specific patterns in text.
- **nltk.corpus**: Provides a list of common words like 'in', 'the', and 'to' that are usually removed to focus on more meaningful words.
- **nltk.stem.porter**: Used for stemming, which reduces words to their basic form, treating words like 'running' and 'run' as the same.
- **sklearn.feature_extraction.text**: Converts text into numerical values, representing the importance of each word across documents.
- **sklearn.model_selection**: Helps in splitting the data into training and testing sets.
- **sklearn.linear_model**: Provides the Logistic Regression algorithm for prediction.
- **sklearn.metrics**: Used for evaluating the accuracy of the model.

## Workflow

1. **Data Import and Exploration**: The dataset is imported, and initial exploration is conducted to understand its structure and content.
2. **Text Preprocessing**: 
   - Removal of stopwords using NLTK.
   - Stemming of words using the PorterStemmer.
3. **Feature Extraction**: 
   - Applying `TfidfVectorizer` to convert text data into numerical form.
4. **Data Splitting**: 
   - Using `train_test_split` to divide the dataset into training and testing sets.
5. **Model Training**: 
   - Training the Logistic Regression model on the training data.
6. **Model Evaluation**: 
   - Predicting on the test data and calculating the accuracy score.

## Conclusion

This project demonstrates the application of machine learning techniques to predict fake news. The use of natural language processing and logistic regression provides a robust framework for handling and analyzing text data.

## Future Work

- Experimenting with other machine learning models like Random Forest, SVM, etc.
- Enhancing text preprocessing with techniques like lemmatization.
- Exploring different feature extraction methods like Word2Vec, GloVe, etc.
