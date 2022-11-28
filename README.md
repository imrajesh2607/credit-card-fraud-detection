# Credit Card Fraud Detection

## Objective
The Objective of this project is to predict whether a credit card transaction is fraudulent or not, based on the transaction amount, location and other transaction related data. It aims to track down credit card transaction data, which is done by detecting anomalies in the transaction data. Credit card fraud detection is typically implemented using an algorithm that detects any anomalies in the transaction data and notifies the cardholder (as a precautionary measure) and the bank about any suspicious transaction.


## Problem Statement
The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning model of Logistic Regression.
In this project, we will analyse customer-level data which has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group.

## Data Source
The dataset is taken from the Kaggle Website website and it has a total of 2,84,807 transactions, out of which 492 are fraudulent.

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Data Description
The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.

## Website
http://cc-fraud-detection-app.herokuapp.com/

## Details
I have used the Logistic Regression Algorithm of Machine Learning. Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1. Logistic regression uses the concept of predictive modeling as regression; therefore, it is called logistic regression, but is used to classify samples; Therefore, it falls under the classification algorithm. 

## Output
0 --> Normal Transaction

1 --> fraudulent transaction
